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
#define T_END 0.01                  // Final simulation time (s)
#define DT_MOVIE 4e-5                  // Video sampling interval (s)
#define DT_PROFILES 1e-3             // 1D profile extraction interval (s)

// Geometry & AMR 
#define X_LENGTH 25e-3              // Domain width (m)
#define Y_LENGTH 25e-3              // Domain height (m)
#define MAX_LEVEL 8                 // Maximum adaptive grid refinement level
#define MIN_LEVEL 4                 // Minimum adaptive grid refinement level
#define PROBE_X (X_LENGTH / 2.0)    // X-position for vertical profile extraction (m)

// Chemistry & Injection 
#define KINFOLDER "C3MechV4_RED.yaml" // Chemical mechanism file
#define V_EVAP 4                      // Injection velocity (m/s)
#define T_WALL 400.0                  // Wall temperature (K)
#define ERATIO 1.0                    // Equivalence ratio
#define V_side 0.0                   // Wind velocity


// Spark 
#define SPARK 1                     // 1: Enable, 0: Disable
#define SPARK_X 2e-3                // Spark center X (m)
#define SPARK_Y 2e-3                // Spark center Y (m)
#define SPARK_DIAM 4e-3             // Spark diameter (m)
#define SPARK_DUR 0.005             // Spark duration (s)
#define SPARK_TEMP 1e7              // Spark temperature (K)

// --- Stoichiometric data ---
#define S_STOICH 2.396                
#define Y_O2_AIR 0.233                // O2 air mass fraction
#define Y_N2_AIR 0.767                // N2 air mass fraction

#define F_FUEL ((ERATIO * Y_O2_AIR) / (S_STOICH + ERATIO * Y_O2_AIR))
#define F_AIR  (1.0 - F_FUEL)
#define Y_FUEL_WALL (F_FUEL)
#define Y_O2_WALL   (Y_O2_AIR * F_AIR)
#define Y_N2_WALL   (Y_N2_AIR * F_AIR)

scalar fuel_old[];
scalar Sd_field[];

// - - - - - - - - - - - - - - - - - - - - - - - -
// Boundary Conditions
// - - - - - - - - - - - - - - - - - - - - - - - -

u.n[bottom] = dirichlet (V_EVAP); 
u.t[bottom] = dirichlet (0.);     
p[bottom]   = neumann (0.);       

u.n[top]    = neumann (0.);
u.t[top]    = neumann (0.);
p[top]      = dirichlet (0.);

u.n[left]   = dirichlet (V_side);
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

// - - - - - - - - - - - - - - - - - - - - - - - -
// Parameters log 
// - - - - - - - - - - - - - - - - - - - - - - - -

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


// - - - - - - - - - - - - - - - - - - - - - - - -
// Initializing
// - - - - - - - - - - - - - - - - - - - - - - - -

event init (i = 0) {
  if (!restore (file = "restart")) {
    foreach() { u.x[] = 0.; u.y[] = 0.; }
  } else {
    restored = true;
  }

  if (pid() == 0) {
    system("rm -f ./flame_v2/*.dat");
    system("rm -f ./flame_v2/*.png");
    system("rm -f ./flame_v2/*.h5");
    system("rm -f ./flame_v2/*.mp4");
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

  scalar fuel = gas->YList[index_species ("IC3H7OH")]; // Anciennement "H2"
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

  foreach() fuel_old[] = fuel[];
}

// - - - - - - - - - - - - - - - - - - - - - - - -
// Logs
// - - - - - - - - - - - - - - - - - - - - - - - -

event print_log (i += 10) {
  double max_T = 0.;
  foreach(reduction(max:max_T)) { 
    if (gas->T[] > max_T) max_T = gas->T[]; 
  }
  
  if (pid() == 0) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[9]; // format "HH:MM:SS"
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);
    fprintf (stderr, "[%s] Ite: %d | t: %e | dt: %e | Cells: %ld | T_max: %.2f K\n", 
             buffer, i, t, dt, grid->tn, max_T);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - -
// Trying to compute flame speed 
// - - - - - - - - - - - - - - - - - - - - - - - -

event compute_flame_speed (i++) {
  scalar fuel = gas->YList[index_species ("IC3H7OH")];
  foreach() {
    double grad_x = (fuel[1,0] - fuel[-1,0]) / (2.*Delta);
    double grad_y = (fuel[0,1] - fuel[0,-1]) / (2.*Delta);
    double grad_f = sqrt(sq(grad_x) + sq(grad_y)) + 1e-12; 
    
    if (grad_f > 1.0) { 
      double dcdt = (fuel[] - fuel_old[]) / dt;
      double u_dot_grad = u.x[] * grad_x + u.y[] * grad_y;
      Sd_field[] = (dcdt + u_dot_grad) / grad_f;
    } else {
      Sd_field[] = 0.;
    }
  }
  foreach() fuel_old[] = fuel[];
}

event output_sd (t += DT_PROFILES; t <= T_END) {
  char filename[80];
  sprintf (filename, "flame_speed_t_%.4f.dat", t);
  FILE * fp = fopen (filename, "w");
  
  scalar fuel = gas->YList[index_species ("IC3H7OH")];
  
  foreach() {
    // Calcul du module du gradient spatial : |grad(Y_fuel)|
    double grad_f = sqrt(sq((fuel[1,0]-fuel[-1,0])/(2.*Delta)) + sq((fuel[0,1]-fuel[0,-1])/(2.*Delta)));
    
    // Écriture conditionnelle dans la zone du front de flamme
    if (grad_f > 1.0) {
      fprintf (fp, "%g %g %g\n", x, y, Sd_field[]);
    }
  }
  fclose (fp);
}

// - - - - - - - - - - - - - - - - - - - - - - - -
// Video Output
// - - - - - - - - - - - - - - - - - - - - - - - -

event movie (t += DT_MOVIE; t <= T_END) {         
  clear(); 
  view (tx = -0.5, ty = -0.5);
  squares ("T", min = 300, max = 5000, linear = true); 
  save ("temperature_evolution.mp4"); 
}


// - - - - - - - - - - - - - - - - - - - - - - - -
// Logging temperature field
// - - - - - - - - - - - - - - - - - - - - - - - -

event temperature_profile (t += DT_PROFILES; t <= T_END) {
  FILE * fp = NULL;
  
  if (pid() == 0) {
    char filename[80];
    sprintf (filename, "temp_profile_full_t_%.4f.dat", t);
    fp = fopen (filename, "w");
    fprintf (fp, "# x (m) | y (m) | T (K)\n");
  }
  
  for (double x_p = 0.; x_p <= X_LENGTH; x_p += X_LENGTH/200.) {
    for (double y_p = 0.; y_p <= Y_LENGTH; y_p += Y_LENGTH/100.) {
      double T_p = interpolate (gas->T, x_p, y_p);
      
      if (pid() == 0 && T_p != nodata) {
        fprintf (fp, "%e %e %e\n", x_p, y_p, T_p);
      }
    }
  }
  
  if (pid() == 0) {
    fclose (fp);
  }
}


// - - - - - - - - - - - - - - - - - - - - - - - -
// Adaptative mesh
// - - - - - - - - - - - - - - - - - - - - - - - -

#if TREE 
event adapt (i++) {
  scalar fuel = gas->YList[index_species ("IC3H7OH")];
  adapt_wavelet ({fuel, gas->T, u.x, u.y},
      (double[]){2e-2, 5e0, 2e-1, 2e-1}, MAX_LEVEL, MIN_LEVEL);
}
#endif
