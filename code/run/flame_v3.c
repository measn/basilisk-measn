// ===========================================================================
// Simulation de Flamme 2D - Basilisk + Cantera (Hydrogène)
// ===========================================================================

#include "navier-stokes/low-mach.h" 
#include "navier-stokes/perfs.h"    
#include "cantera/properties.h"     
#include "cantera/chemistry.h"      
#include "combustion.h"             
#include "gravity.h"                
#include "view.h"                   
#include <time.h>
#include <stdlib.h>

// --- Paramètres Globaux originaux ---
#define T_END 0.005                  
#define DT_MOVIE 1e-5                  
#define DT_PROFILES 1e-5             
#define X_LENGTH 30e-3              
#define Y_LENGTH 30e-3              
#define MAX_LEVEL 8                 
#define MIN_LEVEL 4                 
#define KINFOLDER "laminarflame.yaml" 
#define V_EVAP 0.05                    
#define T_WALL 400.0                  
#define F_FUEL 1.0                    
#define SPARK_X (X_LENGTH / 2.0)
#define SPARK_Y 5e-3
#define ERATIO 1.0

// ===========================================================================
// VARIABLES GLOBALES POUR LES FRONTIÈRES ET L'INJECTION (A PLACER ICI !)
// ===========================================================================
#define T_INJ 300.0
double Y_INJ[150]; // Assez grand pour stocker les fractions massiques

// Fonction magique pour les frontières (Doit être déclarée avant tout event)
double get_Y_INJ (scalar s) {
  int k = s.i - gas->YList[0].i; 
  return Y_INJ[k];
}                


// --- Déclarations globales ---
#ifndef _COMBUSTION_H_
  scalar hrr[], omega_fuel[], omega_o2[], omega_h2o[], omega_oh[];
#endif
scalar fuel_old[]; // Celle-ci reste, car elle est propre à votre simulation

// --- Paramètres de Lissage Géométrique ---
#define X_CENTER (X_LENGTH / 2.0)
#define R_INJ 2.0e-3      
#define DELTA_BC 0.5e-3   
#define SMOOTH_JET(x) (0.5 * (1.0 - tanh((fabs(x - X_CENTER) - R_INJ) / DELTA_BC)))

// ===========================================================================
// CORRECTION DES PROPRIÉTÉS THERMO (Surcharge de la sandbox)
// ===========================================================================

double custom_gasprop_density (void * p) {
  ThermoState * ts = (ThermoState *)p;
  
  // -- L'AIRBAG : Protège Cantera des crashs numériques --
  double T_safe = max(ts->T, 250.0); // Empêche T de passer sous 250K
  T_safe = min(T_safe, 5000.0);      // Empêche l'infini
  
  thermo_setTemperature (thermo, T_safe);
  thermo_setPressure (thermo, ts->P);
  size_t ns = thermo_nSpecies (thermo);
  thermo_setMassFractions (thermo, ns, ts->x, 1); 
  return thermo_density (thermo); 
}

double custom_gasprop_viscosity (void * p) {
  ThermoState * ts = (ThermoState *)p;
  thermo_setTemperature (thermo, ts->T);
  thermo_setPressure (thermo, ts->P);
  size_t ns = thermo_nSpecies (thermo);
  thermo_setMassFractions (thermo, ns, ts->x, 1);
  return trans_viscosity (tran);
}

double custom_gasprop_heatcapacity (void * p) {
  ThermoState * ts = (ThermoState *)p;
  thermo_setTemperature (thermo, ts->T);
  thermo_setPressure (thermo, ts->P);
  size_t ns = thermo_nSpecies (thermo);
  
  thermo_setMassFractions (thermo, ns, ts->x, 1);
  return thermo_cp_mass (thermo);
}

double custom_gasprop_thermalconductivity (void * p) {
  ThermoState * ts = (ThermoState *)p;
  thermo_setTemperature (thermo, ts->T);
  thermo_setPressure (thermo, ts->P);
  size_t ns = thermo_nSpecies (thermo);
  
  thermo_setMassFractions (thermo, ns, ts->x, 1);
  return trans_thermalConductivity (tran); // Utilisation de l'objet 'tran'
}

// --- Conditions aux limites ---
u.n[bottom] = dirichlet( V_EVAP * SMOOTH_JET(x) ); 
u.t[bottom] = dirichlet( 0. );     
p[bottom]   = neumann( 0. );       
pf[bottom]  = neumann( 0. );

u.n[top] = neumann(0.); // Laisse le gaz sortir librement
u.t[top] = neumann(0.); // Laisse glisser
p[top]   = dirichlet(0.); // Fixe la pression relative à 0
pf[top]  = dirichlet(0.);

// WALL / SYMMETRY (Gauche)
u.n[left]   = dirichlet (0.);   
u.t[left]   = neumann (0.);     
p[left]     = neumann (0.);     
pf[left]    = neumann (0.);

u.n[right]  = dirichlet(0.); 
u.t[right]  = neumann(0.);   
p[right]    = neumann(0.);
pf[right]   = neumann(0.);    

int main() {
  kinetics (KINFOLDER);
  NS = thermo_nSpecies (thermo); 
  size (X_LENGTH);
  init_grid (1 << MIN_LEVEL);
  TOLERANCE = 1e-5; 
  run();
}

event stability (i++) {
  if (i < 20) {
    // Les 20 premières itérations : on rampe pour passer le choc initial
    dtmax = 1e-7; 
  } else {
    // Régime de croisière pour l'hydrogène
    dtmax = 5e-6; 
  }
}

event defaults (i = 0) {
  // Surcharge des pointeurs de la sandbox d'Eduardo
  tp2.rhov    = custom_gasprop_density;
  tp2.muv     = custom_gasprop_viscosity;
  tp2.cpv     = custom_gasprop_heatcapacity;
  tp2.lambdav = custom_gasprop_thermalconductivity;
}



// ===========================================================================
// DOMAIN INITIALIZATION & ITERATIVE MESH REFINEMENT
// ===========================================================================
event init (t = 0) {
  size_t ns = thermo_nSpecies (thermo);
  double Y_frais[ns], Y_brules[ns];

  // 1. Compute fresh gases state (Stoichiometric H2/Air)
  thermo_setTemperature (thermo, T_INJ);
  thermo_setPressure (thermo, 101325.0);
  thermo_setMoleFractionsByName (thermo, "H2:1.0, O2:0.5, N2:1.88");
  thermo_getMassFractions (thermo, ns, Y_frais);
  thermo_getMassFractions (thermo, ns, Y_INJ);

  // 2. Compute burnt gases state using Cantera's equilibrium solver (HP)
  thermo_equilibrate (thermo, "HP", 0, 1e-9, 50000, 1000, 0);
  double T_brules = thermo_temperature (thermo);
  thermo_getMassFractions (thermo, ns, Y_brules);

  if (pid() == 0) {
    printf("--- CANTERA INITIALIZATION OK ---\n");
    printf("Adiabatic Flame Temperature: %g K\n", T_brules);
    printf("---------------------------------\n");
  }

  // 3. Set up dynamic boundary conditions for the inlet
  gas->T[bottom] = dirichlet(T_INJ);
  for (int k = 0; k < ns; k++) {
    scalar Yk = gas->YList[k];
    Yk[bottom] = dirichlet( get_Y_INJ(_s) );
  }

  // 4. Flame ball configuration and iterative grid refinement
  double r_noyau = 2.0e-3; // Radius of the burnt gas core (2 mm)
  double delta   = 0.6e-3; // Slightly wider smoothing profile for stable initialization
  int i_h2 = index_species("H2");

  // Iterative loop: Project the profile and refine the mesh step-by-step
  for (int r_iter = 0; r_iter < 6; r_iter++) {
    foreach() {
      double r = sqrt(sq(x - SPARK_X) + sq(y - SPARK_Y));
      
      // Analytical tanh profile for smooth transition
      double alpha = 0.5 * (1.0 - tanh((r - r_noyau) / (delta / 2.0)));

      // Interpolate Temperature and Species fields
      gas->T[] = T_INJ + (T_brules - T_INJ) * alpha;
      for (int k = 0; k < ns; k++) {
        scalar Yk = gas->YList[k];
        Yk[] = Y_frais[k] + (Y_brules[k] - Y_frais[k]) * alpha;
      }
    }
    // Update ghost cells after modifying the internal fields
    boundary (all);

    // Dynamic AMR at t=0: Force Basilisk to capture the steep gradients
    if (i_h2 >= 0) {
      scalar T_loc = gas->T;
      scalar Yh2_loc = gas->YList[i_h2];
      
      // Refine if Temp error > 10K or H2 mass fraction error > 0.01
      adapt_wavelet ({T_loc, Yh2_loc}, (double[]){10.0, 0.01}, MAX_LEVEL, MIN_LEVEL);
    }
  }
}


// Parameters log
event log_parameters (i = 0) {
  if (pid() == 0) {
    FILE * fp = fopen ("parameters.log", "w");
    fprintf (fp, "--- Simulation Parameters ---\n");
    fprintf (fp, "Domain: %g x %g m\n", X_LENGTH, Y_LENGTH);
    fprintf (fp, "End Time: %g s\n", T_END);
    fprintf (fp, "Mechanism: %s\n", KINFOLDER);
    fprintf (fp, "Injection Velocity: %f m/s\n", V_EVAP);
    fprintf (fp, "Equivalence Ratio (Phi): %g\n", ERATIO);
    fprintf (fp, "Max Level: %d\n", MAX_LEVEL);
    fprintf (fp, "Min Level: %d\n", MIN_LEVEL);
    fprintf (fp, "Spark Position: (%g, %g) m\n", SPARK_X, SPARK_Y);
    fclose (fp);
  }
}

// ASCII logs to monitor
event print_log (i += 10) {
  double max_T = -1e30, min_T = 1e30;
  double max_hrr = -1e30, min_hrr = 1e30;

  long local_cells = 0;
  foreach(serial) {
    local_cells++;
  }

  foreach(reduction(max:max_T) reduction(min:min_T)
          reduction(max:max_hrr) reduction(min:min_hrr)) {
    if (gas->T[] > max_T) max_T = gas->T[];
    if (gas->T[] < min_T) min_T = gas->T[];
    if (hrr[] > max_hrr) max_hrr = hrr[];
    if (hrr[] < min_hrr) min_hrr = hrr[];
  }

  long min_cells = local_cells;
  long max_cells = local_cells;

#if _MPI
  MPI_Reduce(&local_cells, &min_cells, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_cells, &max_cells, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
#endif

  if (pid() == 0) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[9];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);

    fprintf (stderr,
             "[%s] Ite: %d | t: %e | dt: %e | Cells/core [min-max]: %ld-%ld (Tot: %ld) | T [min-max]: %.2f-%.2f K | HRR [min-max]: %.2e-%.2e W/m³\n",
             buffer, i, t, dt, min_cells, max_cells, grid->tn, min_T, max_T, min_hrr, max_hrr);
  }
}

// Compute heat release rate and production rates
event compute_hrr (t += DT_PROFILES; t <= T_END) {
  double MWGs[NS];
  molecular_weights(NS, MWGs);

  int idx_fuel = index_species ("H2");
  int idx_o2   = index_species ("O2");
  int idx_h2o  = index_species ("H2O");
  int idx_oh   = index_species ("OH");

  foreach() {
    double Y_local[NS];
    double wdot_kmol[NS];
    double h_molar[NS];

    for (int k = 0; k < NS; k++) {
      scalar Y_k = gas->YList[k];
      Y_local[k] = Y_k[];
    }

    #if _OPENMP
    #pragma omp critical
    #endif
    {
      thermo_setTemperature (thermo, gas->T[]);
      thermo_setPressure (thermo, Pref);
      thermo_setMassFractions (thermo, NS, Y_local, 1);
      kin_getNetProductionRates (kin, NS, wdot_kmol);
      thermo_getPartialMolarEnthalpies (thermo, NS, h_molar);
    }

    if (idx_fuel >= 0) omega_fuel[] = wdot_kmol[idx_fuel] * MWGs[idx_fuel];
    if (idx_o2 >= 0)   omega_o2[]   = wdot_kmol[idx_o2]   * MWGs[idx_o2];
    if (idx_h2o >= 0)  omega_h2o[]  = wdot_kmol[idx_h2o]  * MWGs[idx_h2o];
    if (idx_oh >= 0)   omega_oh[]   = wdot_kmol[idx_oh]   * MWGs[idx_oh];

    double local_hrr = 0.;
    for (int k = 0; k < NS; k++) {
      local_hrr -= wdot_kmol[k] * h_molar[k];
    }
    hrr[] = local_hrr;
  }
}

// VTU output function
void output_pvtu (scalar * slist, vector * vlist, char * prefix, double t) {
  if (pid() == 0) {
    system("mkdir -p vtu/vtk_pieces");
  }
#if _MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  char vtu_name[128];
  char pvtu_name[128];

  sprintf (vtu_name, "vtu/vtk_pieces/%s_t_%.4f_n%d.vtu", prefix, t, pid());
  sprintf (pvtu_name, "vtu/%s_t_%.4f.pvtu", prefix, t);

  long num_cells = 0;
  foreach (serial) { num_cells++; }

  FILE * fp = fopen (vtu_name, "w");
  if (fp) {
    fprintf (fp, "<?xml version=\"1.0\"?>\n");
    fprintf (fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf (fp, "  <UnstructuredGrid>\n");
    fprintf (fp, "    <Piece NumberOfPoints=\"%ld\" NumberOfCells=\"%ld\">\n", num_cells * 4, num_cells);

    fprintf (fp, "      <Points>\n");
    fprintf (fp, "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    foreach (serial) {
      double hs = Delta / 2.0;
      fprintf (fp, "          %g %g 0\n", x - hs, y - hs);
      fprintf (fp, "          %g %g 0\n", x + hs, y - hs);
      fprintf (fp, "          %g %g 0\n", x + hs, y + hs);
      fprintf (fp, "          %g %g 0\n", x - hs, y + hs);
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "      </Points>\n");

    fprintf (fp, "      <Cells>\n");
    fprintf (fp, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    for (long i = 0; i < num_cells * 4; i += 4)
      fprintf (fp, "          %ld %ld %ld %ld\n", i, i+1, i+2, i+3);
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    for (long i = 1; i <= num_cells; i++)
      fprintf (fp, "          %ld\n", i * 4);
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for (long i = 0; i < num_cells; i++)
      fprintf (fp, "          9\n");
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "      </Cells>\n");

    fprintf (fp, "      <CellData>\n");
    for (scalar s in slist) {
      fprintf (fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", s.name);
      foreach (serial) fprintf (fp, "          %g\n", s[]);
      fprintf (fp, "        </DataArray>\n");
    }
    for (vector v in vlist) {
      fprintf (fp, "        <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"ascii\">\n", v.x.name);
      foreach (serial) fprintf (fp, "          %g %g 0\n", v.x[], v.y[]);
      fprintf (fp, "        </DataArray>\n");
    }
    fprintf (fp, "      </CellData>\n");
    fprintf (fp, "    </Piece>\n");
    fprintf (fp, "  </UnstructuredGrid>\n");
    fprintf (fp, "</VTKFile>\n");
    fclose (fp);
  }

  if (pid() == 0) {
    FILE * fpvtu = fopen (pvtu_name, "w");
    if (fpvtu) {
      fprintf (fpvtu, "<?xml version=\"1.0\"?>\n");
      fprintf (fpvtu, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
      fprintf (fpvtu, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
      fprintf (fpvtu, "    <PPoints>\n");
      fprintf (fpvtu, "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\"/>\n");
      fprintf (fpvtu, "    </PPoints>\n");
      fprintf (fpvtu, "    <PCellData>\n");
      for (scalar s in slist)
        fprintf (fpvtu, "      <PDataArray type=\"Float64\" Name=\"%s\" format=\"ascii\"/>\n", s.name);
      for (vector v in vlist)
        fprintf (fpvtu, "      <PDataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"ascii\"/>\n", v.x.name);
      fprintf (fpvtu, "    </PCellData>\n");
      for (int i = 0; i < npe(); i++)
        fprintf (fpvtu, "    <Piece Source=\"vtk_pieces/%s_t_%.4f_n%d.vtu\"/>\n", prefix, t, i);
      fprintf (fpvtu, "  </PUnstructuredGrid>\n");
      fprintf (fpvtu, "</VTKFile>\n");
      fclose (fpvtu);
      fprintf (stderr, "Export PVTU AMR multi-cœurs complété : %s\n", pvtu_name);
    }
  }
}

// ==========================================================
// OUTPUT VTU
// ==========================================================

event snapshot_vtu (t += DT_PROFILES; t <= T_END) {
  char prefix[80];
  sprintf (prefix, "fields"); 

  // 1. Résolution des indices
  int i_h2  = index_species("H2");
  int i_o2  = index_species("O2");
  int i_oh  = index_species("OH");
  int i_h2o = index_species("H2O");

  // 2. Encapsulation conditionnelle (Remplace le 'return;' fautif)
  if (i_h2 >= 0 && i_o2 >= 0 && i_oh >= 0 && i_h2o >= 0) {
    
    // 3. Contournement de l'erreur de parsing '{' : 
    // On initialise la liste uniquement avec les scalaires statiques
    scalar * champs_scalaires = list_copy ({gas->T, p, hrr});
    
    // Ajout séquentiel des scalaires dynamiques Cantera
    champs_scalaires = list_append (champs_scalaires, gas->YList[i_h2]);
    champs_scalaires = list_append (champs_scalaires, gas->YList[i_o2]);
    champs_scalaires = list_append (champs_scalaires, gas->YList[i_oh]);
    champs_scalaires = list_append (champs_scalaires, gas->YList[i_h2o]);

    // 4. Allocation C standard pour le vecteur
    vector * champs_vectoriels = malloc (2 * sizeof(vector));
    champs_vectoriels[0] = u;
    champs_vectoriels[1].x.i = -1; // Marqueur de fin d'itérateur Basilisk

    // 5. Appel de l'export
    output_pvtu (champs_scalaires, champs_vectoriels, prefix, t);

    // 6. Nettoyage mémoire
    free (champs_scalaires);
    free (champs_vectoriels);
  } else {
    // Message de diagnostic optionnel au lieu de l'arrêt brutal
    if (pid() == 0) {
      fprintf(stderr, "[Avertissement] Espèces non trouvées. Export VTU ignoré à t=%g.\n", t);
    }
  }
}


// ==========================================================
// OUTPUT DUMP
// ==========================================================
event snapshot_dump (t += DT_PROFILES; t <= T_END) {
  // Utilisation de mkdir -p pour éviter les erreurs si le dossier existe
  if (pid() == 0) {
    system("mkdir -p dumps"); 
  }
#if _MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  char name[80];
  sprintf (name, "dumps/dump_t_%.4f", t);

  // Initialisation de la liste avec les scalaires natifs
  scalar * export_list = list_copy ({gas->T, p, u.x, u.y, hrr, omega_fuel, omega_oh});

  // Résolution et ajout sécurisé des champs d'espèces
  int i_h2  = index_species("H2");
  int i_o2  = index_species("O2");
  int i_h2o = index_species("H2O");
  int i_oh  = index_species("OH");

  if (i_h2 >= 0)  export_list = list_append (export_list, gas->YList[i_h2]);
  if (i_o2 >= 0)  export_list = list_append (export_list, gas->YList[i_o2]);
  if (i_h2o >= 0) export_list = list_append (export_list, gas->YList[i_h2o]);
  if (i_oh >= 0)  export_list = list_append (export_list, gas->YList[i_oh]);

  dump (name, list = export_list);
  free (export_list); 
  
  if (pid() == 0) fprintf(stderr, "Dump saved: %s\n", name);
}

#if TREE
event adapt (i++) {
  int i_h2 = index_species("H2");
  
  // Encapsulation conditionnelle stricte (pas de return;)
  if (i_h2 >= 0) {
    scalar T_gas = gas->T;
    scalar Y_fuel = gas->YList[i_h2];
    
    adapt_wavelet ({T_gas, Y_fuel, u.x, u.y},
                   (double[]){10.0, 0.05, 1e-3, 1e-3}, MAX_LEVEL, MIN_LEVEL);
  }
}
#endif