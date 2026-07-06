/**
 * 2D Laminar Flame Simulation - Architecture Base
 * * This code serves as the foundation for a robust, adaptive, 
 * and chemically reacting flow simulation.
 */

#include "navier-stokes/low-mach.h" // Low-Mach solver
#include "navier-stokes/perfs.h"    // Performance tracking
#include "cantera/properties.h"     // Cantera properties
#include "cantera/chemistry.h"      // Cantera kinetics
#include "combustion.h"             // Species transport & reactions
#include "gravity.h"                // Buoyancy
#include "view.h"                   // 2D rendering
#include <time.h>
#include <stdlib.h>

// --- Physical and Domain Parameters ---
#define DOMAIN_SIZE     30e-3        // Domain width/height (meters)
#define MAX_LEVEL       10            // Maximum grid refinement level
#define MIN_LEVEL       8            // Minimum grid refinement level

// --- Simulation Constants ---
#define T_END           0.005         // Final time (seconds)
#define CFL_MAX         0.1         // Stability criterion
#define P_ATM           101325.0     // Atmospheric pressure (Pa)
#define T_INITIAL       300.0        // Initial gas temperature (K)

// --- Simulation Parameters ---
#define V_INJ 0.05                  // Injection speed (m/s) 
#define F_FUEL 1                     // Fuel Ratio
// --- Chemistry Mechanism ---
// Note: Ensure the path is relative to the execution directory 
// or define the full absolute path.
#define KIN_MECHANISM   "GRI30_RED.yaml"

// Global solver state
int maxlevel, minlevel = MIN_LEVEL;

// ----------------------------------------------
// --- Conditions aux limites ---
// ----------------------------------------------

// Bottom: Inflow
u.n[bottom] = dirichlet( V_INJ ); 
u.t[bottom] = dirichlet( 0. );     
p[bottom]   = neumann( 0. );       
pf[bottom]  = neumann( 0. );

// Top: Outflow
u.n[top]    = neumann( 0. );       // Sortie libre
u.t[top]    = neumann( 0. );       // Sortie libre
p[top]      = dirichlet( 0. );     // Pression ambiante fixée
pf[top]     = dirichlet( 0. );

// Left: Wall
u.n[left]   = dirichlet( 0. );     // Non-pénétration
u.t[left]   = dirichlet( 0. );     // Non-glissement
p[left]     = neumann( 0. );       // Paroi imperméable
pf[left]    = neumann( 0. );

// Right: Wall
u.n[right]  = dirichlet( 0. );     // Non-pénétration
u.t[right]  = dirichlet( 0. );     // Non-glissement
p[right]    = neumann( 0. );       // Paroi imperméable
pf[right]   = neumann( 0. );



// ----------------------------------------------
// --- Outils Thermodynamiques ---
// ----------------------------------------------
void sanitize_fractions (scalar * YList) {
    foreach() {
        double sum = 0.;
        // 1. Clamping (Bornes physiques strictes)
        for (scalar Y in YList) {
            if (Y[] < 1e-10) Y[] = 1e-10; 
            if (Y[] > 1.0)   Y[] = 1.0;
            sum += Y[];
        }
        // 2. Normalisation sécurisée
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

    // 1. Initialisation de Cantera (Indispensable avant le chargement du CSV)
    // On doit connaître NS et les index des espèces pour mapper les colonnes
    kinetics(KIN_MECHANISM, &NS);
    if (NS <= 0) {
        fprintf(stderr, "FATAL: NS is %d. Kinetics failed.\n", NS);
        return 1;
    }

    gas_species = new_species_names(NS);
    fprintf(stderr, "DEBUG: Species allocated, NS=%d\n", NS);

    // 2. Chargement du profil de flamme (Approche hybride)
    // On charge le CSV une fois en mémoire avant le lancement de la simulation
    load_flame_csv("flame_profile_ch4.csv");

    // 3. Configuration de la grille et des paramètres numériques
    origin(0., 0.);
    size(DOMAIN_SIZE);

    G = (coord){0., -9.81, 0.}; // Gravité selon Y négatif
    CFL = 0.2; 
    Pref = P_ATM;
    T0   = T_INITIAL;

    NITERMAX = 150;
    TOLERANCE = 1e-3; 

    init_grid(1 << MIN_LEVEL);

    // 4. Lancement de la boucle temporelle
    run();

    // 5. Nettoyage mémoire (Cleanup)
    // Libération des données du profil 1D
    if (flame_prof.x != NULL) free(flame_prof.x);
    if (flame_prof.T != NULL) free(flame_prof.T);
    if (flame_prof.u != NULL) free(flame_prof.u);
    if (flame_prof.Y != NULL) {
        for(int s = 0; s < NS; s++) {
            if (flame_prof.Y[s] != NULL) free(flame_prof.Y[s]);
        }
        free(flame_prof.Y);
    }

    // Libération des espèces Cantera
    free_species_names(NS, gas_species);
    
    return 0;
}


// ----------------------------------------------
// --- Initialisation ---
// ----------------------------------------------

event init (i = 0) {
  // 1. Initialisation de base des champs hydrodynamiques
  scalar T = gas->T;
  scalar * YList = gas->YList;

  // ===================================================================
  // --- TOGGLE : CONFIGURATION DU GAZ DE FOND ET INJECTION ---
  // ===================================================================
  // 0 : Gaz frais du CSV (Cohérence parfaite avec la flamme)
  // 1 : Prémélange CH4/Air Stoechiométrique (Phi = 1.0) partout
  // 2 : Air pur dans le domaine, Injection de CH4 pur par le bas
  int bg_config = 2; 

  double T_bg;
  double Y_bg[NS];
  for (int s = 0; s < NS; s++) Y_bg[s] = 0.0;

  // Recherche dynamique des indices des espèces
  int iCH4 = -1, iO2 = -1, iN2 = -1;
  for (int s = 0; s < NS; s++) {
      if (strcmp(gas_species[s], "CH4") == 0) iCH4 = s;
      if (strcmp(gas_species[s], "O2")  == 0) iO2 = s;
      if (strcmp(gas_species[s], "N2")  == 0) iN2 = s;
  }

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
  else {
      T_bg = flame_prof.T[0];
      for (int s = 0; s < NS; s++) {
          Y_bg[s] = flame_prof.Y[s][0];
      }
  }

  // Initialisation globale
  foreach() { 
      T[] = T_bg;
      for (int s = 0; s < NS; s++) {
          scalar Y = YList[s];
          Y[] = Y_bg[s];
      }
      u.x[] = 0.; 
      u.y[] = 0.;
      p[]   = 0.;
  }
  boundary((scalar *){u.x, u.y, p});

  // 2. Initialisation des gradients (sécurisé avec accolades)
  for (scalar s in YList) {
      s.gradient = minmod2;
  }
  T.gradient = minmod2;

  // ===================================================================
  // --- NOUVEAU : CONDITIONS AUX LIMITES DYNAMIQUES POUR L'INJECTION --
  // ===================================================================
  if (bg_config == 1 || bg_config == 2) {
      T[bottom] = dirichlet(300.0); 
      
      for (int s = 0; s < NS; s++) {
          scalar Y = YList[s];
          
          if (bg_config == 2) {
              // CORRECTION : Utilisation stricte des accolades
              if (s == iCH4) {
                  Y[bottom] = dirichlet(1.0);
              } else {
                  Y[bottom] = dirichlet(0.0);
              }
          } 
          else if (bg_config == 1) {
              // CORRECTION : Utilisation stricte des accolades
              if (s == iCH4) {
                  Y[bottom] = dirichlet(0.05518);
              } else if (s == iO2) {
                  Y[bottom] = dirichlet(0.22016);
              } else if (s == iN2) {
                  Y[bottom] = dirichlet(0.72466);
              } else {
                  Y[bottom] = dirichlet(0.0);
              }
          }
      }
  }
  // ===================================================================

  // 3. Overlay local : On "peint" la demi-sphère de flamme
  double R_flame = 2.5e-3; 
  double center_y = 8e-3; 

  foreach() {
    double r = sqrt(sq(x) + sq(y - center_y));
    
    if (r <= R_flame && x >= 0.0) {
      double x_csv = R_flame - r;
      
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


// =================================================================
// --- ADAPTATION DE MAILLAGE (Optimisé pour CH4) ---
// =================================================================
event adapt (i++) {
    scalar * list = NULL;
    
    // 1. On adapte toujours sur le gradient de Température
    list = list_append(list, gas->T);
    
    // 2. Identification dynamique des espèces clés pour CH4
    int iCH4 = index_species("CH4");
    int iOH  = index_species("OH"); // Excellent marqueur de zone de réaction
    int iCO  = index_species("CO"); // Marqueur de la zone d'oxydation secondaire
    
    if (iCH4 >= 0) list = list_append(list, gas->YList[iCH4]);
    if (iOH >= 0)  list = list_append(list, gas->YList[iOH]);
    if (iCO >= 0)  list = list_append(list, gas->YList[iCO]);
    
    int num_scalars = list_len(list);
    double thresholds[num_scalars];
    
    // 3. Définition des métriques (Tolérances absolues)
    int idx = 0;
    thresholds[idx++] = 15.0;  // Température: raffinement si Delta T > 15 K
    
    // Les fractions massiques (Y) évoluent entre 0 et 1.
    if (iCH4 >= 0) thresholds[idx++] = 1e-3; // Le CH4 est majoritaire
    if (iOH >= 0)  thresholds[idx++] = 1e-5; // Le OH est présent en trace (pic ~0.003)
    if (iCO >= 0)  thresholds[idx++] = 1e-4; // Le CO est intermédiaire
    
    adapt_wavelet (list, thresholds, maxlevel = MAX_LEVEL, minlevel = MIN_LEVEL);
                 
    free (list);
}

// =================================================================
// --- UNIFIED MONITORING, DIAGNOSTICS & DIVERGENCE CHECK ---
// =================================================================

event logfile (i += 10) { 
    
    // 1. Initialisation des variables de réduction
    double T_max = -HUGE, T_min = HUGE;
    double P_max = -HUGE, P_min = HUGE;
    double div_max = -HUGE, div_min = HUGE;
    double sum_Y = 0.0;
    
    // 2. Boucle de réduction : Calcul des métriques physiques
    // On ne compte plus les cellules ici manuellement pour éviter l'erreur de compilateur
    foreach(reduction(max:T_max) reduction(min:T_min) 
            reduction(max:P_max) reduction(min:P_min)
            reduction(max:div_max) reduction(min:div_min) 
            reduction(+:sum_Y)) {
        
        // --- Température ---
        if (gas->T[] > T_max) T_max = gas->T[];
        if (gas->T[] < T_min) T_min = gas->T[];
        
        // --- Pression ---
        if (p[] > P_max) P_max = p[];
        if (p[] < P_min) P_min = p[];
        
        // --- Divergence locale ---
        double div = 0.0;
        foreach_dimension() {
            div += (u.x[1] - u.x[-1]) / (2.0 * Delta);
        }
        if (div > div_max) div_max = div;
        if (div < div_min) div_min = div;

        // --- Somme des fractions massiques ---
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
        
        // Moyenne globale de la somme des fractions (doit tendre vers 1.0)
        double avg_Y = sum_Y / (double)grid->tn; 
        
        // Affichage consolidé :
        // grid->tn = total global de cellules
        // grid->n  = cellules sur le cœur courant (PID 0 ici)
        fprintf(stderr, 
            "[%s] i: %-5d | t: %-8.4g | dt: %-8.4g | Cells(Tot/Loc): %ld / %d | T(min/max): %.2f / %.2f K | P(min/max): %.0f / %.0f Pa | Div: %e / %e | SumY: %.4f | mgp: %d / %g\n", 
            time_buffer, i, t, dt, grid->tn, grid->n, T_min, T_max, P_min, P_max, div_min, div_max, avg_Y, mgp.i, mgp.resa);
        
        fflush(stderr);
    }
}

// =================================================================
// --- EXPORT PARALLELE VTK (Corrections structurelles AMR) ---
// =================================================================
#include <sys/stat.h> // Pour mkdir

// Fonction utilitaire pour nettoyer les noms d'espèces (remplace les parenthèses)
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

event snapshot_vtu (t += 0.0001; t <= T_END) {
    
    // 1. Création du dossier par le processus maitre
    if (pid() == 0) {
        mkdir("vtk_pieces", 0777); 
        mkdir("vtu", 0777);
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    // 2. Définition des compteurs LOCAUX
    // L'instruction "serial" indique à Basilisk de ne PAS réduire 
    // cette variable sur le réseau MPI. Elle restera strictement locale.
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
    
    return 0; // Conformité avec l'architecture Basilisk
}