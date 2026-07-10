// 2D Laminar Flame Simulation 
 
#include "navier-stokes/low-mach.h" // Low-Mach solver
#include "navier-stokes/perfs.h"    // Performance tracking
#include "cantera/properties.h"     // Cantera properties
#include "cantera/chemistry.h"      // Cantera kinetics
#include "combustion.h"             // Species transport & reactions
#include "gravity.h"                // Buoyancy
#include "view.h"                   // 2D rendering
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h> 


// --- Physical and Domain Parameters ---
#define DOMAIN_SIZE     60e-3        // Domain width/height (meters)
#define MAX_LEVEL       12            // Maximum grid refinement level
#define MIN_LEVEL       7            // Minimum grid refinement level

// --- Simulation Constants ---
#define T_END           0.5         // Final time (seconds)
#define CFL_MAX         0.2         // Stability criterion
#define P_ATM           101325.0     // Atmospheric pressure (Pa)
#define T_INITIAL       300.0        // Initial gas temperature (K)

// --- Simulation Parameters ---
#define V_INJ 0.1                  // Injection speed (m/s) 
#define F_FUEL 1                     // Fuel Ratio
#define T_RESIDENCE 0.003
#define DT 0.0001
#define Y_flame 1e-3
#define X_FLAME 2e-3
#define R_FLAME 1e-3
// --- Chemistry Mechanism ---
#define KIN_MECHANISM   "GRI30_RED.yaml"

// ===================================================================
// --- Paramètres de Stratification (Config 3) ---
// ===================================================================
#define PHI_BOTTOM 10.0     
#define PHI_TOP    0.0      
#define Y_STRAT    50e-3    

// --- valeurs pour les conditions aux limites ---
#define MASS_TOT_B (16.04 * (PHI_BOTTOM) + 274.654)
#define Y_CH4_B    ((16.04 * (PHI_BOTTOM)) / MASS_TOT_B)
#define Y_O2_B     (63.996 / MASS_TOT_B)
#define Y_N2_B     (210.658 / MASS_TOT_B)


// Déclaration du champ scalaire pour le Heat Release Rate
scalar HRR[];
scalar wdot_CH4[];
scalar S_local[]; // Vitesse de flamme locale (cm/s)

// Global solver state
int maxlevel, minlevel = MIN_LEVEL;

  // ===================================================================
  // --- TOGGLE GLOBAL : CONFIGURATION DU GAZ DE FOND ET INJECTION ---
  // ===================================================================
  // 0 : Gaz frais du CSV (Cohérence parfaite avec la flamme)
  // 1 : Prémélange CH4/Air Stoechiométrique (Phi = 1.0) partout
  // 2 : Air pur dans le domaine, Injection de CH4 pur par le bas
  // 3 : Prémélange stratifié (Phi de 10 en bas à 0 en haut)

int bg_config = 2; 

// ===================================================================
// --- TOGGLE GLOBAL : CONFIGURATION DE LA FLAMME D'ALLUMAGE ---
// ===================================================================
// 0 : Demi-sphère 
// 1 : Rectiligne verticale 
int ign_config = 0; 

// --- CONDITIONS AUX LIMITES ---

u.n[bottom] = dirichlet( bg_config == 2 ? V_INJ : 0. ); 
u.t[bottom] = dirichlet( 0. );     
p[bottom]   = neumann( 0. );       
pf[bottom]  = neumann( 0. );

// Top: Outflow
u.n[top]    = neumann( 0. );       
u.t[top]    = neumann( 0. );       
p[top]      = dirichlet( 0. );   
pf[top]     = dirichlet( 0. );

// Left: Wall
u.n[left]   = dirichlet( 0. );     
u.t[left]   = dirichlet( 0. );    
p[left]     = neumann( 0. );       
pf[left]    = neumann( 0. );

// Right: Wall
u.n[right]  = dirichlet( 0. );     
u.t[right]  = dirichlet( 0. );     
p[right]    = neumann( 0. );       
pf[right]   = neumann( 0. );

// ----------------------------------------------
// --- Outils Thermodynamiques ---
// ----------------------------------------------
void sanitize_fractions (scalar * YList) {
    foreach() {
        double sum = 0.;
        for (scalar Y in YList) {
            if (Y[] < 1e-10) Y[] = 1e-10; 
            if (Y[] > 1.0)   Y[] = 1.0;
            sum += Y[];
        }
        for (scalar Y in YList) {
            Y[] /= sum;
        }
    }
    // 3. Frontières (Crucial pour le solveur NS)
    boundary (YList);
}

// =================================================================
// --- STRUCTURES ET DONNEES DE LA FLAMME 1D ---
// =================================================================

// Structure pour stocker le profil précalculé
typedef struct {
    int n_points;
    double *x;
    double *T;
    double *u;
    double **Y; // Matrice des fractions massiques Y[index_espece][index_point]
} FlameProfile;

FlameProfile flame_prof;

// Fonction d'interpolation linéaire avec maintien aux bornes (Clamping)
double interpolate_1D(double x_target, double * x_array, double * y_array, int n) {
    if (n == 0) return 0.0;
    if (x_target <= x_array[0]) return y_array[0];
    if (x_target >= x_array[n-1]) return y_array[n-1];
    
    // Recherche de l'intervalle
    int i = 0;
    while (i < n - 1 && x_array[i+1] < x_target) {
        i++;
    }
    
    double dx = x_array[i+1] - x_array[i];
    if (dx < 1e-12) return y_array[i]; // Sécurité contre la division par zéro
    
    double t = (x_target - x_array[i]) / dx;
    return y_array[i] + t * (y_array[i+1] - y_array[i]);
}

////// structure csv
void load_flame_csv(const char * filename) {
    FILE * fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERREUR CRITIQUE: Impossible d'ouvrir le profil %s\n", filename);
        exit(1);
    }
    
    // 1. Compter le nombre de lignes pour allouer la mémoire
    int lines = 0;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), fp)) lines++;
    flame_prof.n_points = lines - 1; // Exclusion de l'en-tête
    
    // 2. Allocation mémoire
    flame_prof.x = (double *)malloc(flame_prof.n_points * sizeof(double));
    flame_prof.T = (double *)malloc(flame_prof.n_points * sizeof(double));
    flame_prof.u = (double *)malloc(flame_prof.n_points * sizeof(double));
    flame_prof.Y = (double **)malloc(NS * sizeof(double *));
    
    for (int s = 0; s < NS; s++) {
        flame_prof.Y[s] = (double *)malloc(flame_prof.n_points * sizeof(double));
        for (int i = 0; i < flame_prof.n_points; i++) flame_prof.Y[s][i] = 0.0;
    }
    
    // 3. Lecture de l'en-tête et mapping des colonnes
    rewind(fp);
    fgets(buffer, sizeof(buffer), fp);
    
    int col_mapping[500]; // Map: index_colonne -> index_espece (-1 si non espèce, -2 pour T, -3 pour x)
    for(int i=0; i<500; i++) col_mapping[i] = -1;
    
    int col_idx = 0;
    char * token = strtok(buffer, ", \n\r");
    while (token != NULL) {
        if (strcmp(token, "x") == 0) col_mapping[col_idx] = -3;
        else if (strcmp(token, "T") == 0) col_mapping[col_idx] = -2;
        else if (strncmp(token, "Y_", 2) == 0) {
            // Extraction du nom de l'espèce (ex: "Y_CH4" -> "CH4")
            int s_idx = index_species(token + 2); 
            if (s_idx != -1) col_mapping[col_idx] = s_idx;
        }
        col_idx++;
        token = strtok(NULL, ", \n\r");
    }
    
    // 4. Extraction des données
    int pt = 0;
    while (fgets(buffer, sizeof(buffer), fp) && pt < flame_prof.n_points) {
        col_idx = 0;
        token = strtok(buffer, ", \n\r");
        while (token != NULL) {
            double val = atof(token);
            if (col_mapping[col_idx] == -3) flame_prof.x[pt] = val;
            else if (col_mapping[col_idx] == -2) flame_prof.T[pt] = val;
            else if (col_mapping[col_idx] >= 0) flame_prof.Y[col_mapping[col_idx]][pt] = val;
            
            col_idx++;
            token = strtok(NULL, ", \n\r");
        }
        pt++;
    }
    fclose(fp);
    
    if (pid() == 0) {
        printf("INFO: Profil de flamme charge avec succes (%d points).\n", flame_prof.n_points);
    }
}

// ----------------------------------------------
// --- Main ---
// ----------------------------------------------

int main (int argc, char ** argv) {

    kinetics(KIN_MECHANISM, &NS);
    if (NS <= 0) {
        fprintf(stderr, "FATAL: NS is %d. Kinetics failed.\n", NS);
        return 1;
    }

    gas_species = new_species_names(NS);
    fprintf(stderr, "DEBUG: Species allocated, NS=%d\n", NS);

    load_flame_csv("flame_profile_ch4.csv");
    origin(0., 0.);
    size(DOMAIN_SIZE);

    G = (coord){0., -9.81, 0.}; 
    CFL = 0.2; 
    Pref = P_ATM;
    T0   = T_INITIAL;

    NITERMAX = 150;
    TOLERANCE = 1e-2; 

    init_grid(1 << MIN_LEVEL);

    run();

    if (flame_prof.x != NULL) free(flame_prof.x);
    if (flame_prof.T != NULL) free(flame_prof.T);
    if (flame_prof.u != NULL) free(flame_prof.u);
    if (flame_prof.Y != NULL) {
        for(int s = 0; s < NS; s++) {
            if (flame_prof.Y[s] != NULL) free(flame_prof.Y[s]);
        }
        free(flame_prof.Y);
    }

    free_species_names(NS, gas_species);

    return 0;
}


// ----------------------------------------------
// --- Initialisation ---
// ----------------------------------------------

event init_0 (i = 0) {
  // 1. Initialisation de base des champs hydrodynamiques
  scalar T = gas->T;
  scalar * YList = gas->YList;

  double T_bg = T_INITIAL;
  double Y_bg[NS];
  for (int s = 0; s < NS; s++) Y_bg[s] = 0.0;

  // Recherche dynamique des indices des espèces
  int iCH4 = -1, iO2 = -1, iN2 = -1;
  for (int s = 0; s < NS; s++) {
      if (strcmp(gas_species[s], "CH4") == 0) iCH4 = s;
      if (strcmp(gas_species[s], "O2")  == 0) iO2 = s;
      if (strcmp(gas_species[s], "N2")  == 0) iN2 = s;
  }

  // Préparation des configurations uniformes (0, 1 et 2)
  if (bg_config == 1) {
      T_bg = 300.0; 
      if (iCH4 != -1) Y_bg[iCH4] = 0.05518;
      if (iO2  != -1) Y_bg[iO2]  = 0.22016;
      if (iN2  != -1) Y_bg[iN2]  = 0.72466;
  } 
  else if (bg_config == 2) {
      T_bg = 300.0;
      if (iO2  != -1) Y_bg[iO2]  = 0.233;
      if (iN2  != -1) Y_bg[iN2]  = 0.767;
  }
  else if (bg_config == 0) {
      T_bg = flame_prof.T[0];
      for (int s = 0; s < NS; s++) {
          Y_bg[s] = flame_prof.Y[s][0];
      }
  }

  // Initialisation globale sur le maillage
  foreach() { 
      if (bg_config == 3) {
            T[] = 300.0;
            
            double phi;
            if (y <= Y_STRAT) {
                phi = PHI_BOTTOM - (PHI_BOTTOM - PHI_TOP) * (y / Y_STRAT);
            } 
            else {
                phi = PHI_TOP;
            }
            
            if (phi < 0.0) phi = 0.0;
            
            double mass_tot = 16.04 * phi + 274.654;
            
            for (int s = 0; s < NS; s++) {
                scalar Y = YList[s];
                Y[] = 0.0;
            }
            if (iCH4 != -1) { scalar Y_CH4 = YList[iCH4]; Y_CH4[] = (16.04 * phi) / mass_tot; }
            if (iO2  != -1) { scalar Y_O2  = YList[iO2];  Y_O2[]  = 63.996 / mass_tot; }
            if (iN2  != -1) { scalar Y_N2  = YList[iN2];  Y_N2[]  = 210.658 / mass_tot; }
      } else {
          // --- CONFIGURATIONS UNIFORMES ---
          T[] = T_bg;
          for (int s = 0; s < NS; s++) {
              scalar Y = YList[s];
              Y[] = Y_bg[s];
          }
      }
      
      // Vitesses et pression initiales
      u.x[] = 0.; 
      u.y[] = 0.;
      p[]   = 0.;
  } // Fin du foreach proprement fermée

  boundary((scalar *){u.x, u.y, p});

  // 2. Initialisation des gradients (sécurisé avec accolades)
  for (scalar s in YList) {
      s.gradient = minmod2;
  }
  T.gradient = minmod2;

  if (bg_config == 1 || bg_config == 2 || bg_config == 3) {
      T[bottom] = dirichlet(300.0); 
      
      for (int s = 0; s < NS; s++) {
          scalar Y = YList[s];
          
          if (bg_config == 2) {
              if (s == iCH4) { Y[bottom] = dirichlet(1.0); } 
              else { Y[bottom] = dirichlet(0.0); }
          } 
          else if (bg_config == 1) {
              if (s == iCH4) { Y[bottom] = dirichlet(0.05518); } 
              else if (s == iO2) { Y[bottom] = dirichlet(0.22016); } 
              else if (s == iN2) { Y[bottom] = dirichlet(0.72466); } 
              else { Y[bottom] = dirichlet(0.0); }
          }
          else if (bg_config == 3) {
              if (s == iCH4) { 
                  Y[bottom] = dirichlet(Y_CH4_B); 
              } 
              else if (s == iO2) { 
                  Y[bottom] = dirichlet(Y_O2_B); 
              } 
              else if (s == iN2) { 
                  Y[bottom] = dirichlet(Y_N2_B); 
              } 
              else { 
                  Y[bottom] = dirichlet(0.0); 
              }
          }
      } 
  } 
  
  // 4. Overlay local : On "peint" la flamme d'allumage
  double R_flame = R_FLAME; 
  double center_y = Y_flame; 

  foreach() {
      bool in_flame = false;
      double x_csv = 0.0;

      // Choix de la géométrie
      if (ign_config == 0) {
          // Config 0 : Demi-sphère (R_flame est ici le RAYON)
          double r = sqrt(sq(x) + sq(y - center_y));
          if (r <= R_flame && x >= 0.0) {
              in_flame = true;
              x_csv = R_flame - r;
          }
      } 
      else if (ign_config == 1) {
          // Config 1 : Flamme rectiligne verticale
          // ON REMPLACE R_flame PAR X_FLAME
          if (x <= X_FLAME && x >= 0.0) {
              in_flame = true;
              x_csv = X_FLAME - x; // Le front est maintenant à X_FLAME
          }
      }

      // Application du profil si on est dans la zone d'allumage
      if (in_flame) {
          T[] = interpolate_1D(x_csv, flame_prof.x, flame_prof.T, flame_prof.n_points);
          for (int s = 0; s < NS; s++) {
              scalar Y = YList[s];
              Y[] = interpolate_1D(x_csv, flame_prof.x, flame_prof.Y[s], flame_prof.n_points);
          }
      }
  }

  // 4. Clôture et nettoyage
  boundary({T});
  boundary(YList);
  sanitize_fractions(YList); 
  
  event("properties"); 

  // 5. Calcul initial du volume spécifique alpha (1/rho)
  foreach() {
    if (rho[] < 0.01) {
        rho[] = 1.0; 
    }
  }
  foreach_face() {
      double rhof = (rho[] + rho[-1]) / 2.0;
      alpha.x[] = 1.0 / rhof; 
  }
  boundary ((scalar *){alpha.x, alpha.y});
}

// -------------------------------------------------------------------
// --- ÉVÉNEMENT : Temps de résidence de la flamme ---
// -------------------------------------------------------------------
event flame_residence (t <= T_RESIDENCE) {
  scalar T = gas->T;
  scalar * YList = gas->YList;

  double R_flame = R_FLAME;
  double center_y = Y_flame; 

  foreach() {
        bool in_flame = false;
        double x_csv = 0.0;

        if (ign_config == 0) {
            // Config 0 : Demi-sphère 
            double r = sqrt(sq(x) + sq(y - center_y));
            if (r <= R_flame && x >= 0.0) {
                in_flame = true;
                x_csv = R_flame - r;
            }
        } 
        else if (ign_config == 1) {
            // Config 1 : Flamme rectiligne verticale
            if (x <= X_FLAME && x >= 0.0) {
                in_flame = true;
                x_csv = X_FLAME - x; 
            }
        }

        // Application du profil si on est dans la zone d'allumage
        if (in_flame) {
            T[] = interpolate_1D(x_csv, flame_prof.x, flame_prof.T, flame_prof.n_points);
            for (int s = 0; s < NS; s++) {
                scalar Y = YList[s];
                Y[] = interpolate_1D(x_csv, flame_prof.x, flame_prof.Y[s], flame_prof.n_points);
            }
        }
    }

  // 1. Clôture et nettoyage des frontières MPI
  boundary({T});
  boundary(YList);
  sanitize_fractions(YList);
  
  // 2. On force Cantera à recalculer la densité (rho) et la viscosité
  event("properties"); 

  // 3. On met à jour le volume spécifique (alpha) pour Navier-Stokes
  foreach() {
    if (rho[] < 0.01) {
        rho[] = 1.0; 
    }
  }
  foreach_face() {
      double rhof = (rho[] + rho[-1]) / 2.0;
      alpha.x[] = 1.0 / rhof; 
  }
  boundary ((scalar *){alpha.x, alpha.y});
}

// =================================================================
// --- ADAPTATION DE MAILLAGE (Optimisé pour CH4) ---
// =================================================================
event adapt (i++) {
    scalar * list = NULL;
    
    // 1. Thermique ET Hydrodynamique (Crucial pour les instabilités)
    list = list_append(list, gas->T);
    list = list_append(list, u.x);
    list = list_append(list, u.y);
    
    // 2. Identification dynamique des espèces clés
    int iCH4 = index_species("CH4");
    int iOH  = index_species("OH");  // Marqueur de la zone de réaction (très fin)
    int iCO  = index_species("CO");  // Marqueur des gaz brûlés
    
    if (iCH4 >= 0) list = list_append(list, gas->YList[iCH4]);
    if (iOH >= 0)  list = list_append(list, gas->YList[iOH]);
    if (iCO >= 0)  list = list_append(list, gas->YList[iCO]);
    
    int num_scalars = list_len(list);
    double thresholds[num_scalars];
    
    // 3. Définition des métriques (Tolérances absolues)
    int idx = 0;
    thresholds[idx++] = 5.0;   // T : Plus strict (5K) pour ne pas lisser le front
    thresholds[idx++] = 0.02;  // u.x : Raffiner si la vitesse varie de 2 cm/s
    thresholds[idx++] = 0.02;  // u.y : Raffiner si la vitesse varie de 2 cm/s
    
    // Tolérances chimiques (tes valeurs étaient parfaites)
    if (iCH4 >= 0) thresholds[idx++] = 1e-3; 
    if (iOH >= 0)  thresholds[idx++] = 1e-5; 
    if (iCO >= 0)  thresholds[idx++] = 1e-4; 
    
    adapt_wavelet (list, thresholds, maxlevel = MAX_LEVEL, minlevel = MIN_LEVEL);
                 
    free (list);
}

event compute_hrr (i++) {
    int ns = NS; 
    
    // 1. Alias locaux AVANT la boucle pour éviter les bugs de qcc
    scalar * YList = gas->YList;
    scalar T_gas = gas->T;
    
    // On récupère l'index du CH4 une seule fois avant la boucle (optimisation)
    int idx_CH4 = index_species("CH4"); 
    
    foreach() {
        double ymass[ns];
        double hm[ns];    // Enthalpies molaires partielles [J/kmol]
        double wdot[ns];  // Taux de production [kmol/m^3/s]
        
        // 2. Lecture propre des fractions massiques locales
        for (int s = 0; s < ns; s++) {
            scalar Y = YList[s];
            ymass[s] = Y[]; 
        }
        
        // 3. Appel de l'API Cantera pour récupérer la thermo et la cinétique
        thermo_setTemperature(thermo, T_gas[]);
        thermo_setPressure(thermo, P_ATM); 
        thermo_setMassFractions(thermo, ns, ymass, 1);
        
        thermo_getPartialMolarEnthalpies(thermo, ns, hm); 
        kin_getNetProductionRates(kin, ns, wdot);
        
        // 4. Calcul du HRR en W/m^3
        double hrr_local = 0.0;
        for (int s = 0; s < ns; s++) {
            hrr_local -= wdot[s] * hm[s];
        }
        HRR[] = hrr_local;
        
        // 5. Calcul du taux de production/consommation massique du CH4 (kg/m3/s)
        // wdot est en kmol/m3/s, on le multiplie par la masse molaire (16.04 kg/kmol)
        if (idx_CH4 >= 0) {
            wdot_CH4[] = wdot[idx_CH4] * 16.04;
        } else {
            wdot_CH4[] = 0.0;
        }
    }
    
    // 6. Mise à jour des frontières MPI pour les deux champs
    boundary({HRR, wdot_CH4}); 
}

// =================================================================
// --- Monitoring ---
// =================================================================

event logfile (i += 10) { 
    
    // 1. Initialisation des variables de réduction
    double T_max = -HUGE, T_min = HUGE;
    double P_max = -HUGE, P_min = HUGE;
    double div_max = -HUGE, div_min = HUGE;
    double sum_Y = 0.0;
    
    foreach(reduction(max:T_max) reduction(min:T_min) 
            reduction(max:P_max) reduction(min:P_min)
            reduction(max:div_max) reduction(min:div_min) 
            reduction(+:sum_Y)) {
        
        if (gas->T[] > T_max) T_max = gas->T[];
        if (gas->T[] < T_min) T_min = gas->T[];

        if (p[] > P_max) P_max = p[];
        if (p[] < P_min) P_min = p[];
        
        double div = 0.0;
        foreach_dimension() {
            div += (u.x[1] - u.x[-1]) / (2.0 * Delta);
        }
        if (div > div_max) div_max = div;
        if (div < div_min) div_min = div;
        for (scalar s in gas->YList) sum_Y += s[];
    }

    // 3. Output sur le processus maître (MPI rank 0)
    if (pid() == 0) {
        
        time_t rawtime;
        struct tm * timeinfo;
        char time_buffer[80];
        
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        
        double avg_Y = sum_Y / (double)grid->tn; 
        
        fprintf(stderr, 
            "[%s] i: %-5d | t: %-8.4g | dt: %-8.4g | Cells(Tot/Loc): %ld / %d | T(min/max): %.2f / %.2f K | P(min/max): %.0f / %.0f Pa | Div: %e / %e | SumY: %.4f | mgp: %d / %g\n", 
            time_buffer, i, t, dt, grid->tn, grid->n, T_min, T_max, P_min, P_max, div_min, div_max, avg_Y, mgp.i, mgp.resa);
        
        fflush(stderr);
    }
}

// =================================================================
// --- flame speed monitoring
// =================================================================
event monitor_speed (i += 10) { 
    double sum_wdot = 0.0; 
    double A_front = 0.0; 
    
    // Obligatoire pour calculer un gradient spatial propre
    boundary({gas->T}); 

    foreach(reduction(+:sum_wdot) reduction(+:A_front)) {
        // 1. Intégrale de la consommation de CH4
        sum_wdot += wdot_CH4[] * dv();
        
        // 2. Calcul de la surface locale (Théorème de Co-aire)
        double delta_T = 2220.0 - 300.0; 
        double grad_Tx = (gas->T[1,0] - gas->T[-1,0]) / (2.0 * Delta);
        double grad_Ty = (gas->T[0,1] - gas->T[0,-1]) / (2.0 * Delta);
        double grad_c = sqrt(sq(grad_Tx) + sq(grad_Ty)) / delta_T;
        
        A_front += grad_c * dv();
    }

    if (pid() == 0) {
        double rho_0 = 1.16;       
        double Y_CH4_0 = 0.05518;  
        
        double Sc = 0.0;
        if (A_front > 1e-6) {
            Sc = -sum_wdot / (rho_0 * Y_CH4_0 * A_front); 
        }
        
        // Ouverture du fichier en mode "append" (ajout)
        static FILE * fp = fopen("flame_speed.csv", "w");
        if (i == 0) {
            // En-têtes du fichier CSV au premier pas de temps
            fprintf(fp, "time,area_m,speed_cm_s\n");
        }
        // Écriture des données
        fprintf(fp, "%g,%g,%g\n", t, A_front, Sc * 100.0);
        fflush(fp);
    }
}

// =================================================================
// --- CALCUL LOCAL : Vitesse de flamme en chaque point ---
// =================================================================
event compute_local_speed (i++) {
    double rho_0 = 1.16;
    double Y_CH4_0 = 0.05518;
    double delta_T = 2220.0 - 300.0;

    boundary({gas->T, wdot_CH4, HRR});

    foreach() {
        double grad_Tx = (gas->T[1,0] - gas->T[-1,0]) / (2.0 * Delta);
        double grad_Ty = (gas->T[0,1] - gas->T[0,-1]) / (2.0 * Delta);
        double grad_c = sqrt(sq(grad_Tx) + sq(grad_Ty)) / delta_T;

        if (HRR[] > 1e5) { 
            // On s'assure de ne pas diviser par un gradient trop petit
            if (grad_c > 1000.0) {
                S_local[] = 100.0 * (-wdot_CH4[]) / (rho_0 * Y_CH4_0 * grad_c);
            } else {
                S_local[] = 0.0;
            }
        } else {
            S_local[] = 0.0; // En dehors du front, pas de vitesse
        }
    }
    boundary({S_local});
}

// =================================================================
// --- EXPORT PARALLELE VTK ---
// =================================================================

void sanitize_vtk_name(const char * input, char * output) {
    int i = 0;
    while(input[i] != '\0' && i < 255) {
        if(input[i] == '(' || input[i] == ')' || input[i] == '-' || input[i] == '+') {
            output[i] = '_';
        } else {
            output[i] = input[i];
        }
        i++;
    }
    output[i] = '\0';
}

event snapshot_vtu (t += DT; t <= T_END) {
    
    // 1. Création du dossier par le processus maitre
    if (pid() == 0) {
        mkdir("vtk_pieces", 0777); 
        mkdir("vtu", 0777);
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    // 2. Définition des compteurs LOCAUX
    int ncells = 0;
    foreach(serial) {
        ncells++;
    }
    int npoints = ncells * 4; 

    // 3. Ouverture du fichier local
    char name[256];
    sprintf(name, "vtk_pieces/fields_t_%.4f_n%d.vtu", t, pid());
    FILE * fp = fopen(name, "w");
    if (!fp) return 0;

    // --- HEADER XML ---
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <UnstructuredGrid>\n");
    fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", npoints, ncells);

    // --- POINTS ---
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    foreach() {
        double d = Delta / 2.0;
        fprintf(fp, "%.8g %.8g 0.0\n", x - d, y - d); 
        fprintf(fp, "%.8g %.8g 0.0\n", x + d, y - d); 
        fprintf(fp, "%.8g %.8g 0.0\n", x + d, y + d); 
        fprintf(fp, "%.8g %.8g 0.0\n", x - d, y + d); 
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Points>\n");

    // --- CELL DATA ---
    fprintf(fp, "      <CellData>\n");
    
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"T\" format=\"ascii\">\n");
    foreach() fprintf(fp, "%.8g\n", isfinite(gas->T[]) ? gas->T[] : 0.0);
    fprintf(fp, "        </DataArray>\n");
    
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    foreach() fprintf(fp, "%.8g %.8g 0.0\n", isfinite(u.x[]) ? u.x[] : 0.0, isfinite(u.y[]) ? u.y[] : 0.0);
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"HRR\" format=\"ascii\">\n");
    foreach() fprintf(fp, "%.8g\n", HRR[]);
    fprintf(fp, "        </DataArray>\n");  

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"wdot_CH4\" format=\"ascii\">\n");
    foreach() fprintf(fp, "%.8g\n", wdot_CH4[]);
    fprintf(fp, "        </DataArray>\n"); 

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"S_local\" format=\"ascii\">\n");
    foreach() fprintf(fp, "%.8g\n", S_local[]);
    fprintf(fp, "        </DataArray>\n");

    // Espèces chimiques avec noms nettoyés
    for (int s = 0; s < NS; s++) {
        char safe_name[256];
        sanitize_vtk_name(gas_species[s], safe_name);
        
        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", safe_name);
        scalar Y = gas->YList[s];
        foreach() fprintf(fp, "%.8g\n", isfinite(Y[]) ? Y[] : 0.0);
        fprintf(fp, "        </DataArray>\n");
    }
    
    fprintf(fp, "      </CellData>\n"); 

    // --- CONNECTIVITY ---
    fprintf(fp, "      <Cells>\n");
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (int i = 0; i < ncells; i++) {
        fprintf(fp, "%d %d %d %d\n", i*4, i*4+1, i*4+2, i*4+3); 
    }
    fprintf(fp, "        </DataArray>\n");
    
    fprintf(fp, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for (int i = 0; i < ncells; i++) {
        fprintf(fp, "%d\n", (i+1)*4); 
    }
    fprintf(fp, "        </DataArray>\n");
    
    fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for (int i = 0; i < ncells; i++) {
        fprintf(fp, "9\n"); 
    }
    fprintf(fp, "        </DataArray>\n");
    
    fprintf(fp, "      </Cells>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </UnstructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);

    // =================================================================
    // 4. ECRITURE DU FICHIER MAITRE (.pvtu)
    // =================================================================
    if (pid() == 0) {
        char master_name[256];
        sprintf(master_name, "vtu/fields_t_%.4f.pvtu", t);
        FILE * fpvtu = fopen(master_name, "w");
        if (fpvtu) {
            fprintf(fpvtu, "<?xml version=\"1.0\"?>\n");
            fprintf(fpvtu, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
            fprintf(fpvtu, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
            
            fprintf(fpvtu, "    <PPoints>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "    </PPoints>\n");
            
            fprintf(fpvtu, "    <PCellData>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"T\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"HRR\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"wdot_CH4\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"S_local\" format=\"ascii\"/>\n");
            
            // Synchronisation du nommage XML pour le fichier maître
            for (int s = 0; s < NS; s++) {
                char safe_name[256];
                sanitize_vtk_name(gas_species[s], safe_name);
                fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"%s\" format=\"ascii\"/>\n", safe_name);
            }
            fprintf(fpvtu, "    </PCellData>\n");
            
            for (int i = 0; i < npe(); i++) {
                fprintf(fpvtu, "    <Piece Source=\"../vtk_pieces/fields_t_%.4f_n%d.vtu\"/>\n", t, i);
            }
            
            fprintf(fpvtu, "  </PUnstructuredGrid>\n");
            fprintf(fpvtu, "</VTKFile>\n");
            fclose(fpvtu);
        }
    }
    
    return 0; 
}

// Video Output
event movie (t += DT; t <= T_END) {         
  clear(); 
  view (tx = -0.5, ty = -0.5);
  squares ("T", min = 300, max = 2500, linear = true); 
  save ("temperature_evolution.mp4"); 
}