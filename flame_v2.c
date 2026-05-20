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

// Time & Output 
#define T_END 0.015                  // Final simulation time (s)
#define DT_MOVIE 1e-4               // Video sampling interval (s)
#define DT_PROFILES 5e-3            // 1D profile extraction interval (s)

// Geometry & AMR 
#define X_LENGTH 20e-3             // Domain width (m)
#define Y_LENGTH 50e-3              // Domain height (m)
#define MAX_LEVEL 7                 // Maximum adaptive grid refinement level
#define MIN_LEVEL 4                 // Minimum adaptive grid refinement level
#define PROBE_X (X_LENGTH / 2.0)    // X-position for vertical profile extraction (m)

// Chemistry & Injection 
#define KINFOLDER "laminarflame.yaml" // Chemical mechanism file
#define V_EVAP 1                  // Injection velocity (m/s)
#define T_WALL 400.0                // Wall temperature (K)
#define ERATIO 1.0                  // Equivalence ratio

// Spark 
#define SPARK 1                     // 1: Enable, 0: Disable
#define SPARK_X (X_LENGTH / 2.0)    // Spark center X (m)
#define SPARK_Y 1e-3                // Spark center Y (m)
#define SPARK_DIAM 5e-3           // Spark diameter (m)
#define SPARK_DUR 0.002             // Spark duration (s)
#define SPARK_TEMP 1e7              // Spark temperature (K)

// --- Stoichiometric data ---
#define S_STOICH 7.9365               // O2/H2 mass ratio
#define Y_O2_AIR 0.233                // O2 air mass fraction
#define Y_N2_AIR 0.767                // N2 air mass fraction

#define F_FUEL ((ERATIO * Y_O2_AIR) / (S_STOICH + ERATIO * Y_O2_AIR))
#define F_AIR  (1.0 - F_FUEL)
#define Y_FUEL_WALL (F_FUEL)
#define Y_O2_WALL   (Y_O2_AIR * F_AIR)
#define Y_N2_WALL   (Y_N2_AIR * F_AIR)

// --- Boundary Conditions ---
u.n[bottom] = dirichlet (V_EVAP); 
u.t[bottom] = dirichlet (0.);     
p[bottom]   = neumann (0.);       

u.n[top]    = neumann (0.);
u.t[top]    = neumann (0.);
p[top]      = dirichlet (0.);

u.n[left]   = dirichlet (0.);
u.t[left]   = neumann (0.);
p[left]     = neumann (0.);

u.n[right]  = dirichlet (0.);
u.t[right]  = neumann (0.);
p[right]    = neumann (0.);

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

  for (maxlevel = MAX_LEVEL; maxlevel <= MAX_LEVEL; maxlevel++) {
    init_grid (1 << maxlevel); 
    run();
  }

  free_species_names (NS, gas_species), gas_species = NULL;
}


// Log of the parameters
event log_parameters (i = 0) {
  if (pid() == 0) {
    FILE * fp = fopen ("parameters.log", "w");
    fprintf (fp, "--- Simulation Parameters ---\n");
    fprintf (fp, "Domain: %g x %g m\n", X_LENGTH, Y_LENGTH);
    fprintf (fp, "End Time: %g s\n", T_END);
    fprintf (fp, "Mechanism: %s\n", KINFOLDER);
    fprintf (fp, "Injection Velocity: %g m/s\n", V_EVAP);
    fprintf (fp, "Equivalence Ratio (Phi): %g\n", ERATIO);
    fprintf (fp, "Max Level: %d\n", MAX_LEVEL);
    fprintf (fp, "Max Level: %d\n", MIN_LEVEL);
    fprintf (fp, "Spark Position: (%g, %g) m\n", SPARK_X, SPARK_Y);
    fprintf (fp, "Spark Temp: %g K\n", SPARK_TEMP);
    fclose (fp);
    fprintf (stderr, "Parameters logged to parameters.log\n");
  }
}

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

  scalar fuel  = gas->YList[index_species ("H2")]; 
  scalar oxi   = gas->YList[index_species ("O2")]; 
  scalar inert = gas->YList[index_species ("N2")]; 
  scalar T     = gas->T; 

  fuel[bottom]  = dirichlet (Y_FUEL_WALL);
  oxi[bottom]   = dirichlet (Y_O2_WALL);
  inert[bottom] = dirichlet (Y_N2_WALL);
  T[bottom]     = dirichlet (T_WALL);

  fuel[top]     = neumann (0.); 
  oxi[top]      = neumann (0.); 
  inert[top]    = neumann (0.); 
  T[top]        = neumann (0.); 
}

event print_log (i += 10) {
  double max_T = 0.;
  foreach(reduction(max:max_T)) { 
    if (gas->T[] > max_T) max_T = gas->T[]; 
  }
  
  if (pid() == 0) {
    // Récupération de l'heure système
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[9]; // Pour le format "HH:MM:SS"

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);

    // Affichage avec l'heure en début de ligne
    fprintf (stderr, "[%s] Ite: %d | t: %e | dt: %e | Cells: %ld | T_max: %.2f K\n", 
             buffer, i, t, dt, grid->tn, max_T);
  }
}

event movie (t += DT_MOVIE; t <= T_END) {         
  clear(); 
  view (tx = -0.5, ty = -0.25, width = 800, height = 400); 
  squares ("T", min = 300, max = 4000, linear = true); 
  cells(); 
  colorbar (format = "%.0f K", pos = {0.8, -0.8}); 
  save ("temperature_evolution.mp4"); 
}

event vertical_profiles (t += DT_PROFILES; t <= T_END) {
  FILE * fp = NULL;
  
  if (pid() == 0) { 
    char filename[80];
    sprintf (filename, "VerticalProfile_t_%.4f.dat", t); 
    fp = fopen (filename, "w"); 
    fprintf (fp, "# y(m)  T(K)  Y_H2  Y_H2O  u_y(m/s)\n"); 
  }
  
  scalar fuel = gas->YList[index_species ("H2")]; 
  scalar h2o  = gas->YList[index_species ("H2O")]; 

  for (double y_p = 0.; y_p <= Y_LENGTH; y_p += Y_LENGTH/100.) { 
    double T_p = interpolate (gas->T, PROBE_X, y_p); 
    double Y_F = interpolate (fuel, PROBE_X, y_p);   
    double Y_W = interpolate (h2o, PROBE_X, y_p);    
    double V_y = interpolate (u.y, PROBE_X, y_p);    
    
    if (pid() == 0 && T_p != nodata) { 
      fprintf (fp, "%e  %1.4f  %e  %e  %e\n", y_p, T_p, Y_F, Y_W, V_y);
    }
  }
  
  if (pid() == 0) { 
    fclose (fp);
  }
}

#if TREE 
event adapt (i++) {
  scalar fuel = gas->YList[index_species ("H2")];
  // Augmentation des seuils pour limiter le raffinement inutile
  // fuel: 1e-2 -> 2e-2
  // T:    1e0  -> 5e0 (plus tolérant)
  // u:    1e-1 -> 2e-1 (plus tolérant)
  adapt_wavelet ({fuel, gas->T, u.x, u.y},
      (double[]){2e-2, 5e0, 2e-1, 2e-1}, MAX_LEVEL, MIN_LEVEL);
}
#endif

