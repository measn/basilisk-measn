// 2D Laminar Flame

#define SPARK 1 // Enable spark module

#include "navier-stokes/low-mach.h" // Low-Mach solver
#include "navier-stokes/perfs.h"    // Performance tracking
#include "cantera/properties.h"     // Cantera properties
#include "cantera/chemistry.h"      // Cantera kinetics
#include "combustion.h"             // Species transport & reactions
#include "gravity.h"                // Buoyancy
#include "spark.h"                  // Spark ignition
#include "view.h"                   // 2D rendering

// --- Geometry & Time ---
#define X_LENGTH 100e-3             // Domain width (m)
#define Y_LENGTH 50e-3              // Domain height (m)
#define T_END 1.5                   // Simulation end time (s)

// --- Chemistry & Injection ---
#define KINFOLDER "laminarflame.yaml" // Mechanism file
#define V_EVAP 0.5                   // Injection velocity (m/s)
#define T_WALL 400.0                  // Injection temperature (K)

#define ERATIO 1.0                    // Equivalence ratio (Phi)
#define S_STOICH 7.9365               // Mass stoichiometric ratio (O2/H2)
#define Y_O2_AIR 0.233                // Ambient O2 mass fraction
#define Y_N2_AIR 0.767                // Ambient N2 mass fraction

// Wall injection fractions
#define F_FUEL ((ERATIO * Y_O2_AIR) / (S_STOICH + ERATIO * Y_O2_AIR))
#define F_AIR  (1.0 - F_FUEL)

#define Y_FUEL_WALL (F_FUEL)
#define Y_O2_WALL   (Y_O2_AIR * F_AIR)
#define Y_N2_WALL   (Y_N2_AIR * F_AIR)

// --- Boundary Conditions ---
// Bottom: Injection
u.n[bottom] = dirichlet (V_EVAP); 
u.t[bottom] = dirichlet (0.);     
p[bottom]   = neumann (0.);       

// Top: Outflow
u.n[top]    = neumann (0.);
u.t[top]    = neumann (0.);
p[top]      = dirichlet (0.);

// Left: Symmetry
u.n[left]   = dirichlet (0.);
u.t[left]   = neumann (0.);
p[left]     = neumann (0.);

// Right: Symmetry
u.n[right]  = dirichlet (0.);
u.t[right]  = neumann (0.);
p[right]    = neumann (0.);

// --- Global Variables ---
int maxlevel, minlevel = 5; // AMR levels
bool restored = false;      // Restart flag
scalar qspark[];            // Spark energy field

int main (int argc, char ** argv) {
  kinetics (KINFOLDER, &NS);            // Load kinetics
  gas_species = new_species_names (NS); // Alloc species names

  Pref = 101325.; // Reference pressure (Pa)
  T0 = 300.;      // Ambient temperature (K)
  L0 = X_LENGTH;  // Base domain size
  
  G.x = 0.; 
  G.y = -9.81;    // Gravity
  CFL_MAX = 0.1;  // Acoustic stability limit

  for (maxlevel = 8; maxlevel <= 8; maxlevel++) {
    init_grid (1 << maxlevel); // Init 512x512 base grid
    run();
  }

  free_species_names (NS, gas_species), gas_species = NULL;
}

// --- Initialization Event (Runs at t = 0) ---
event init (i = 0) {
  
  // Mask removed to ensure MPI load balancing stability

  // Check for restart file
  if (!restore (file = "restart")) {
    foreach() {
      u.x[] = 0.; // Init horizontal velocity
      u.y[] = 0.; // Init vertical velocity
    }
  } else {
    restored = true; // Mark as restarted
  }

  double x[NS]; // Temp array for mass fractions
  for (int s = 0; s < NS; s++) x[s] = 0.; // Zero all species
  x[index_species ("O2")] = Y_O2_AIR; // Set ambient O2
  x[index_species ("N2")] = Y_N2_AIR; // Set ambient N2

  ThermoState tsg; // Thermo state struct
  tsg.T = T0, tsg.P = Pref, tsg.x = x; // Assign T, P, and fractions
  phase_set_thermo_state (gas, &tsg, force = !restored); // Apply to gas phase

  double MWGs[NS]; // Array for molecular weights
  molecular_weights (NS, MWGs); // Fetch from Cantera
  phase_set_properties (gas, MWs = MWGs); // Apply to gas phase

// --- Spark Setup ---
#if SPARK
  spark.T = qspark; // Link temperature source
  spark.position = (coord){50e-3, 5e-3}; // Set XY center (m)
  spark.diameter = 1.5e-3; // Set spark diameter (m)
  spark.time = 0.; // Start at t=0
  spark.duration = 0.005; // Last for 5ms
  spark.temperature = 1e7; // Target spark temp (K)
  spark.phase = gas; // Apply to gas phase
#endif

  scalar fuel  = gas->YList[index_species ("H2")]; // Fuel pointer
  scalar oxi   = gas->YList[index_species ("O2")]; // O2 pointer
  scalar inert = gas->YList[index_species ("N2")]; // N2 pointer
  scalar T     = gas->T; // Temperature pointer

  // Apply bottom injection conditions
  fuel[bottom]  = dirichlet (Y_FUEL_WALL);
  oxi[bottom]   = dirichlet (Y_O2_WALL);
  inert[bottom] = dirichlet (Y_N2_WALL);
  T[bottom]     = dirichlet (T_WALL);

  // Apply top outflow conditions
  fuel[top]     = neumann (0.); 
  oxi[top]      = neumann (0.); 
  inert[top]    = neumann (0.); 
  T[top]        = neumann (0.); 
}

// --- Terminal Log (MPI safe) ---
event print_log (i += 10) {
  double max_T = 0.;
  foreach(reduction(max:max_T)) { // MPI-safe max reduction
    if (gas->T[] > max_T) max_T = gas->T[]; // Find peak T
  }
  if (pid() == 0) { // Master core only
    fprintf (stderr, "Ite: %d | t: %e | dt: %e | Cells: %ld | T_max: %.2f K\n", 
             i, t, dt, grid->n, max_T);
  }
}

// --- Video Output ---
event movie (t += 5e-3; t <= T_END) {         
  clear(); // Clear drawing buffer
  view (tx = -0.5, ty = -0.25, width = 800, height = 400); // Set camera
  squares ("T", min = 300, max = 4000, linear = true); // Draw T field
  cells(); // Draw mesh edges
  
  // Les valeurs vont généralement de -1 à 1 par rapport au cadre de la caméra.
  colorbar (format = "%.0f K", pos = {0.8, -0.8}); 
  
  save ("temperature_evolution.mp4"); // Save to video
}

// --- 1D Profiles (MPI safe) ---
event vertical_profiles (t += 5e-3; t <= T_END) {
  FILE * fp = NULL;
  
  if (pid() == 0) { // 1. Master opens file
    char filename[80];
    sprintf (filename, "VerticalProfile_t_%.4f.dat", t); // Format name
    fp = fopen (filename, "w"); // Open write mode
    fprintf (fp, "# y(m)  T(K)  Y_H2  Y_H2O  u_y(m/s)\n"); // Write header
  }
  
  scalar fuel = gas->YList[index_species ("H2")]; // H2 pointer
  scalar h2o  = gas->YList[index_species ("H2O")]; // H2O pointer

  for (double y_p = 0.; y_p <= Y_LENGTH; y_p += Y_LENGTH/100.) { // Loop Y axis
    
    // 2. Global MPI interpolation (Executed by ALL cores)
    double T_p = interpolate (gas->T, 1.5e-3, y_p); // Get T
    double Y_F = interpolate (fuel, 1.5e-3, y_p); // Get H2
    double Y_W = interpolate (h2o, 1.5e-3, y_p); // Get H2O
    double V_y = interpolate (u.y, 1.5e-3, y_p); // Get Vy
    
    if (pid() == 0 && T_p != nodata) { // 3. Master writes valid data
      fprintf (fp, "%e  %1.4f  %e  %e  %e\n", y_p, T_p, Y_F, Y_W, V_y);
    }
  }
  
  if (pid() == 0) { // 4. Master closes file
    fclose (fp);
  }
}

// --- Adaptive Mesh Refinement ---
#if TREE // If Quadtree enabled
event adapt (i++) {
  scalar fuel = gas->YList[index_species ("H2")]; // Target fuel grad
  // Refine based on H2, T, and velocity gradients with specific tolerances
  adapt_wavelet ({fuel, gas->T, u.x, u.y},
      (double[]){1e-2, 1e0, 1e-1, 1e-1}, maxlevel, minlevel);
}
#endif