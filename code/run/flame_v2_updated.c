// 2D Laminar Flame

#include "navier-stokes/low-mach.h" // Low-Mach solver
#include "navier-stokes/perfs.h"    // Performance tracking
#include "cantera/properties.h"     // Cantera properties
#include "cantera/chemistry.h"      // Cantera kinetics
#include "combustion.h"             // Species transport & reactions
#include "gravity.h"                // Buoyancy
#include "spark.h"                  // Spark ignition
#include "view.h"                   // 2D rendering
#include <time.h>
#include <stdlib.h>

// Time & Output 
#define T_END 0.005                  // Final simulation time (s)
#define DT_MOVIE 1e-5                  // Video sampling interval (s)
#define DT_PROFILES 1e-5             // 1D profile extraction interval (s)

// Geometry & AMR 
#define X_LENGTH 30e-3              // Domain width (m)
#define Y_LENGTH 30e-3              // Domain height (m)
#define MAX_LEVEL 8                 // Maximum adaptive grid refinement level
#define MIN_LEVEL 4                 // Minimum adaptive grid refinement level

// Chemistry & Injection 
// #define KINFOLDER "C3MechV4_RED.yaml" // Chemical mechanism file for isoprop
#define KINFOLDER "laminarflame.yaml" // Chemical mechanism file
#define V_EVAP 0.05                    // Injection velocity (m/s)
#define T_WALL 400.0                  // Wall temperature (K)
#define ERATIO 1.0                    // Equivalence ratio
#define V_side 1.0                   // Wind velocity

// Spark 
#define SPARK 1                     // 1: Enable, 0: Disable
#define SPARK_X 2e-3                // Spark center X (m)
#define SPARK_Y 2e-3                // Spark center Y (m)
#define SPARK_DIAM 2e-3             // Spark diameter (m)
#define SPARK_DUR 2e-3              // Spark duration (s)
#define SPARK_TEMP 1e5              // Spark temperature (K)

// Stoichiometric data
#define Y_O2_AIR 0.233                // O2 air mass fraction
#define Y_N2_AIR 0.767                // N2 air mass fraction

#define F_FUEL 1

scalar fuel_old[];
scalar Sd_field[];

// scalars for heat release rate and production rates
scalar hrr[];              // Heat Release Rate (W/m³)
scalar omega_fuel[];       // H2 Production Rate (kg/m³/s)
scalar omega_o2[];         // O2 Production Rate (kg/m³/s)
scalar omega_h2o[];        // H2O Production Rate(kg/m³/s)
scalar omega_oh[];         // OH Production Rate (kg/m³/s)


// --- Boundary Conditions ---
// INLET (Bas)
u.n[bottom] = dirichlet (V_EVAP); 
u.t[bottom] = dirichlet (0.);     
p[bottom]   = neumann (0.);       
pf[bottom]  = neumann (0.);

// OUTLET (Haut) - Sortie libre parfaite
u.n[top]    = neumann (0.);
u.t[top]    = neumann (0.);
p[top]      = dirichlet (0.);
pf[top]     = dirichlet (0.); 

// WALL / SYMMETRY (Gauche)
u.n[left]   = dirichlet (0.);   
u.t[left]   = neumann (0.);     
p[left]     = neumann (0.);     
pf[left]    = neumann (0.);

// WALL / SYMMETRY (Droite) - Retour à la normale
u.n[right]  = dirichlet(0.); 
u.t[right]  = neumann(0.);   
p[right]    = neumann(0.);
pf[right]   = neumann(0.);

int maxlevel, minlevel = MIN_LEVEL; 
bool restored = false;              
scalar qspark[];                    

int main (int argc, char ** argv) {
  kinetics (KINFOLDER, &NS);            
  gas_species = new_species_names (NS); 

  Pref = 101325.; 
  T0 = 300.;      
  L0 = X_LENGTH;  
  
  G.x = 0.; 
  G.y = -9.81;    
  CFL_MAX = 0.25;  

  TOLERANCE = 1e-4; 
  NITERMAX = 100;   

  init_grid (1 << MAX_LEVEL); 
  run();
  free_species_names (NS, gas_species), gas_species = NULL;
  
  return 0;
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
    fprintf (fp, "Max Level: %d\n", MIN_LEVEL);
    fprintf (fp, "Spark Position: (%g, %g) m\n", SPARK_X, SPARK_Y);
    fprintf (fp, "Spark Temp: %g K\n", SPARK_TEMP);
    fclose (fp);
  }
}

// Initializing
event init (i = 0) {
  if (!restore (file = "restart")) {
    foreach() { u.x[] = 0.; u.y[] = 0.; }
  } else {
    restored = true;
  }

  double x[NS];
  for (int s = 0; s < NS; s++) x[s] = 0.;
  x[index_species ("O2")] = Y_O2_AIR;
  x[index_species ("N2")] = Y_N2_AIR;

  ThermoState tsg;
  tsg.T = T0, tsg.P = Pref, tsg.x = x;
  phase_set_thermo_state (gas, &tsg, force = !restored);

  double MWGs[NS];
  molecular_weights (NS, MWGs);
  phase_set_properties (gas, MWs = MWGs);

#if SPARK
  spark.T = qspark; 
  spark.position = (coord){SPARK_X, SPARK_Y};
  spark.diameter = SPARK_DIAM; 
  spark.time = 0.; 
  spark.duration = SPARK_DUR; 
  spark.temperature = SPARK_TEMP; 
  spark.phase = gas; 
#endif

  scalar fuel = gas->YList[index_species ("H2")]; 
  scalar oxi  = gas->YList[index_species ("O2")]; 
  scalar inert= gas->YList[index_species ("N2")]; 
  scalar T    = gas->T; 

  fuel[bottom]  = dirichlet (F_FUEL);
  oxi[bottom]   = dirichlet (0.);
  inert[bottom] = dirichlet (0.);
  T[bottom]     = dirichlet (T_WALL);

  fuel[top]     = neumann (0.); 
  oxi[top]      = neumann (0.); 
  inert[top]    = neumann (0.); 
  T[top]        = neumann (0.); 

  foreach() fuel_old[] = fuel[];
  
  // --- CORRECTIONS CRITIQUES POUR t=0 ---
  
  // 1. Appliquer les nouvelles conditions de limites scalaires aux cellules fantômes
  boundary((scalar *){fuel, oxi, inert, T});
  boundary(all);
  
  // 2. Mettre à jour la densité et les propriétés de transport AVANT le solveur
  // Si votre combustion.h possède une routine spécifique, appelez-la ici.
  // Sinon, c'est généralement :
  // update_thermodynamics(); ou équivalent
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

  // 4. Affichage centralisé sur le processus maître
  if (pid() == 0) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[9]; // format "HH:MM:SS"
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);

    fprintf (stderr, 
             "[%s] Ite: %d | t: %e | dt: %e | Cells/core [min-max]: %ld-%ld (Tot: %ld) | T [min-max]: %.2f-%.2f K | HRR [min-max]: %.2e-%.2e W/m³\n", 
             buffer, i, t, dt, min_cells, max_cells, grid->tn, min_T, max_T, min_hrr, max_hrr);
  }
}

// Événement de sécurité thermodynamique (Pur Écrêtage)
// Exécuté à chaque pas de temps pour nettoyer les artefacts de dispersion numérique
event thermodynamic_safeguard (i++) {
  
  // 1. Alias local pour éviter les conflits OpenMP/MPI avec qcc
  scalar T_gas = gas->T;
  
  foreach() {
    // Blocage strict de l'undershoot thermique pour les polynômes NASA
    if (T_gas[] < 300.0) {
      T_gas[] = 300.0;
    }
    // Sécurité haute optionnelle (à ajuster selon votre flamme)
    else if (T_gas[] > 4000.0) {
      T_gas[] = 4000.0;
    }
  }
  // Mise à jour immédiate des cellules fantômes pour la température
  boundary({T_gas});

  // 2. Traitement indépendant des espèces chimiques
  for (int k = 0; k < NS; k++) {
    scalar Yk = gas->YList[k];
    foreach() {
      // Blocage des fractions massiques négatives uniquement
      // AUCUNE normalisation n'est effectuée ici pour préserver la cohérence de rho
      if (Yk[] < 0.0) {
        Yk[] = 0.0;
      }
      else if (Yk[] > 1.0) {
        Yk[] = 1.0;
      }
    }
    boundary({Yk});
  }
}

// Écrêtage et Normalisation (Anti-Gibbs)
event properties (i++) {
  foreach() {
    double sumY = 0.;
    for (scalar Y in gas->YList) {
      Y[] = clamp(Y[], 0.0, 1.0); // Bloque les valeurs négatives
      sumY += Y[];
    }
    if (sumY > 0.) {
      for (scalar Y in gas->YList) Y[] /= sumY; // Force la somme à 1.0
    }
    // Sécurité thermique pour éviter les crashs des polynômes NASA
    gas->T[] = max(gas->T[], 290.0); 
  }
  // Synchronisation des cellules fantômes (Crucial pour l'AMR en multi-cœurs)
  boundary(gas->YList);
  boundary({gas->T});
}

// compute heat release rate and production rates
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

    // ω (kg/m³/s) = wdot (kmol/m³/s) * MW (kg/kmol)
    if (idx_fuel >= 0) omega_fuel[] = wdot_kmol[idx_fuel] * MWGs[idx_fuel];
    if (idx_o2 >= 0)   omega_o2[]   = wdot_kmol[idx_o2]   * MWGs[idx_o2];
    if (idx_h2o >= 0)  omega_h2o[]  = wdot_kmol[idx_h2o]  * MWGs[idx_h2o];
    if (idx_oh >= 0)   omega_oh[]   = wdot_kmol[idx_oh]   * MWGs[idx_oh];

    // HRR = - Σ (ω_k_molar * h_k_molar)
    double local_hrr = 0.;
    for (int k = 0; k < NS; k++) {
      local_hrr -= wdot_kmol[k] * h_molar[k];
    }
    hrr[] = local_hrr;
  }
}

// vtu function 
void output_pvtu (scalar * slist, vector * vlist, char * prefix, double t) {
  if (pid() == 0) {
    mkdir("vtu", 0777); 
    mkdir("vtu/vtk_pieces", 0777); 
  }
  #if _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  char vtu_name[128];
  char pvtu_name[128];
  
  sprintf (vtu_name, "vtu/vtk_pieces/%s_t_%.4f_n%d.vtu", prefix, t, pid());
  sprintf (pvtu_name, "vtu/%s_t_%.4f.pvtu", prefix, t);

  long num_cells = 0;
  foreach (serial) { 
    num_cells++; 
  }
  
  FILE * fp = fopen (vtu_name, "w");
  if (fp) {
    fprintf (fp, "<?xml version=\"1.0\"?>\n");
    fprintf (fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf (fp, "  <UnstructuredGrid>\n");
    fprintf (fp, "    <Piece NumberOfPoints=\"%ld\" NumberOfCells=\"%ld\">\n", num_cells * 4, num_cells);

    // --- AMR Quadrangles ---
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
    for (long i = 0; i < num_cells * 4; i += 4) {
      fprintf (fp, "          %ld %ld %ld %ld\n", i, i + 1, i + 2, i + 3);
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    for (long i = 1; i <= num_cells; i++) {
      fprintf (fp, "          %ld\n", i * 4);
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for (long i = 0; i < num_cells; i++) {
      fprintf (fp, "          9\n");
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "      </Cells>\n");
    
    fprintf (fp, "      <CellData>\n");
    for (scalar s in slist) {
      fprintf (fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", s.name);
      foreach (serial) {
        fprintf (fp, "          %g\n", s[]);
      }
      fprintf (fp, "        </DataArray>\n");
    }
    for (vector v in vlist) {
      fprintf (fp, "        <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"ascii\">\n", v.x.name);
      foreach (serial) {
        fprintf (fp, "          %g %g 0\n", v.x[], v.y[]);
      }
      fprintf (fp, "        </DataArray>\n");
    }
    fprintf (fp, "      </CellData>\n");
    fprintf (fp, "    </Piece>\n");
    fprintf (fp, "  </UnstructuredGrid>\n");
    fprintf (fp, "</VTKFile>\n");
    fclose (fp);
  }

  // writing main file
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
      for (scalar s in slist) {
        fprintf (fpvtu, "      <PDataArray type=\"Float64\" Name=\"%s\" format=\"ascii\"/>\n", s.name);
      }
      for (vector v in vlist) {
        fprintf (fpvtu, "      <PDataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"ascii\"/>\n", v.x.name);
      }
      fprintf (fpvtu, "    </PCellData>\n");

      for (int i = 0; i < npe(); i++) {
        fprintf (fpvtu, "    <Piece Source=\"vtk_pieces/%s_t_%.4f_n%d.vtu\"/>\n", prefix, t, i);
      }
  
      fprintf (fpvtu, "  </PUnstructuredGrid>\n");
      fprintf (fpvtu, "</VTKFile>\n");
      fclose (fpvtu);
      fprintf (stderr, "Export PVTU AMR multi-cœurs complété : %s\n", pvtu_name);
    }
  }
}

// OUTPUT 
event snapshot_dump (t += DT_PROFILES; t <= T_END) {
  if (pid() == 0) {
    mkdir("dumps", 0777); 
  }
  #if _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  char name[80];
  sprintf (name, "dumps/dump_t_%.4f", t);

  scalar * base_list = {gas->T, p, u.x, u.y, hrr, omega_fuel, omega_oh};
  scalar * export_list = list_copy (base_list);

  export_list = list_append (export_list, gas->YList[index_species("H2")]);
  export_list = list_append (export_list, gas->YList[index_species("O2")]);
  export_list = list_append (export_list, gas->YList[index_species("H2O")]);
  export_list = list_append (export_list, gas->YList[index_species("OH")]);

  dump (name, list = export_list);
  free(export_list); 
  
  if (pid() == 0) fprintf(stderr, "Dump saved: %s\n", name);
}

event snapshot_vtu (t += DT_PROFILES; t <= T_END) {
  char prefix[80];
  sprintf (prefix, "fields"); 
  
  scalar fuel = gas->YList[index_species("H2")];
  scalar oxi  = gas->YList[index_species("O2")];
  scalar oh   = gas->YList[index_species("OH")];
  scalar h2o  = gas->YList[index_species("H2O")];

  scalar * champs_scalaires = {gas->T, p, hrr, fuel, oxi, oh, h2o};
  vector * champs_vectoriels = {u};

  output_pvtu (champs_scalaires, champs_vectoriels, prefix, t);
}

// Video Output
event movie (t += DT_MOVIE; t <= T_END) {         
  clear(); 
  view (tx = -0.5, ty = -0.5);
  squares ("T", min = 300, max = 3000, linear = true); 
  save ("temperature_evolution.mp4"); 
}

// Adaptive mesh
#if TREE 
event adapt (i++) {
  // Alias locaux (bonne pratique)
  scalar T_gas = gas->T;
  scalar Y_H2  = gas->YList[index_species("H2")];
  scalar Y_OH  = gas->YList[index_species("OH")];

  // Ajout d'une tolérance adaptative
  // L'idée est de raffiner plus fort sur le front de flamme (OH) 
  // et plus lâchement sur la vitesse pour éviter les instabilités.
  adapt_wavelet ((scalar *){T_gas, Y_H2, Y_OH, u.x, u.y},
                 (double[]){5.0, 5e-4, 5e-5, 0.1, 0.1}, 
                 maxlevel = MAX_LEVEL, 
                 minlevel = MIN_LEVEL);
                 
  // CRUCIAL : Forcer la mise à jour des frontières après l'adaptation
  // car l'AMR redistribue les cellules entre les processus MPI
  boundary(all);
}
#endif

