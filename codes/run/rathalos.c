// Rathalos - 2D monophase Isopropanol flame propagation simulation

/* This code was developed relying heavily on E.Cipriano's work on his laminarflame.c 
and uses includes from his sandbox such as the Cantera wrapper.
This case tries to reproduce the flame runner configuration found at l'Institut Jean Le Rond d'Alembert - Saint-Cyr */
 
#include "navier-stokes/low-mach.h" // Low-Mach solver
#include "navier-stokes/perfs.h"    // Performance tracking
#include "cantera/properties.h"     // Cantera properties
#include "cantera/chemistry.h"      // Cantera kinetics
#include "combustion.h"             // Species transport & reactions
#include "gravity.h"                // Buoyancy
#include "view.h"                   // 2D rendering
#include <time.h>                   // Time module
#include <stdlib.h>
#include <sys/stat.h> 

// --- Kinetics ---
#define KIN_MECHANISM   "C3MechV4_RED.yaml" // Mechanism input
#define FUEL_NAME       "IC3H7OH"           // Exact name of the fuel species in the mechanism

// ===================================================================
// --- CONFIGURATION PANEL ---
// ===================================================================

// BG_CONFIG : Configuration of the fuel injection
// 0 = No fuel
// 1 = Injection of premixed fuel (phi=1) from the bottom
// 2 = Injection of fuel (not mixed) from the bottom
// 3 = Stratified mix (vertical richness gradient)
#define BG_CONFIG 3

// STRAT_PROFILE: Stratification profile configuration
// 0 = Linear, 1 = Exponential, 2 = Logarithmic
#define STRAT_PROFILE 1       

// STRAT_CURVATURE: Tuning parameter for the intensity of the gradient curvature
// Must be strictly positive (e.g., 5.0)
#define STRAT_CURVATURE 5.0

// IGN_CONFIG : Ignition Geometry 
// 0 = Half sphere flame
// 1 = Linear vertical flame
// 2 = Linear horizontal flame
#define IGN_CONFIG 1

// WALL_CONFIG : Boundary conditions (left and right)
// 0 = closed 
// 1 = outflow
#define WALL_CONFIG 1


// --- Physical and Domain Parameters ---
#define DOMAIN_SIZE     40e-3        // Domain width/height (meters)
#define MAX_LEVEL       10           // Maximum grid refinement level
#define MIN_LEVEL       7            // Minimum grid refinement level

// --- Simulation Constants ---
#define T_END           0.080       // Final time (seconds)
#define DT              0.0001       // Log interval
#define CFL_MAX         0.2          // Stability criterion
#define P_ATM           101325.0     // Atmospheric pressure (Pa)
#define T_INITIAL       300.0        // Initial gas temperature (K)

// --- Simulation Parameters ---
#define V_INJ           0.0          // Injection speed (m/s) 
#define F_FUEL          1            // Fuel Ratio
#define T_RESIDENCE     0.0        // Time residency of the flame
#define Y_flame         8e-3         // Position of the ignition flame on the Y axis (meters)
#define X_FLAME         1e-3            // Position of the ignition flame on the X axis (meters)
#define R_FLAME         2e-3         // Radius of the spherical flame (meters)
#define H_FLAME         10e-3

// --- Stratification Parameters (Config 3) ---
#define PHI_BOTTOM 2.5     // Bottom boundary richness
#define PHI_TOP    0     // Richness at Y_STRAT
#define Y_STRAT    10e-3   // height for the stratification

#define AIR_N2_O2_RATIO (0.79 / 0.21) // Molar ratio of N2 to O2 in standard air


// ===================================================================
// --- Misc ---
// ===================================================================

// Various properties and computations
scalar phi_local[];  // Local equivalence ratio field
scalar HRR[];        // Heat Release Rate
scalar wdot_fuel[];  // Fuel consumption rate
scalar S_local[];    // Local flame speed (cm/s)

int maxlevel, minlevel = MIN_LEVEL; // Min and Max level for grid refining

double Y_FUEL_STOICH = 0.0; // Stoichiometric mass fraction for the fuel
double Y_O2_STOICH   = 0.0; // Stoichiometric mass fraction for O2
double Y_N2_STOICH   = 0.0; // Stoichiometric mass fraction for N2
double Y_O2_AIR      = 0.0; // Pure air mass fraction for O2
double Y_N2_AIR      = 0.0; // Pure air mass fraction for N2
double Y_FUEL_B = 0.0;
double Y_O2_B   = 0.0;
double Y_N2_B   = 0.0;

double * atoms_C = NULL;
double * atoms_H = NULL;
double * atoms_O = NULL;
double * MW_array = NULL;

// Helper function to compute mass fractions from the atomic equivalence ratio (Bilger fraction basis)
void get_Y_from_atomic_phi(double phi_target, int iFUEL, int iO2, int iN2, double * Y_res) {
    for (int s = 0; s < NS; s++) {
        Y_res[s] = 0.0;
    }
    
    // Safety check for missing species indices
    if (iFUEL == -1 || iO2 == -1 || iN2 == -1) return;

    double mw_o2 = MW_array[iO2];
    double mw_n2 = MW_array[iN2];

    // Handle pure air boundary (phi -> 0)
    if (phi_target <= 1e-6) {
        double m_tot = mw_o2 + AIR_N2_O2_RATIO * mw_n2;
        Y_res[iO2] = mw_o2 / m_tot;
        Y_res[iN2] = (AIR_N2_O2_RATIO * mw_n2) / m_tot;
        return;
    }
    
    // Retrieve atomic composition of the fuel
    double nC = atoms_C[iFUEL];
    double nH = atoms_H[iFUEL];
    double nO = atoms_O[iFUEL];
    
    // Compute required moles of O2 per mole of fuel
    double X_O2 = (4.0 * nC + nH - 2.0 * nO * phi_target) / (4.0 * phi_target);
    if (X_O2 < 0.0) X_O2 = 0.0; // Cap for extremely rich mixtures
    
    // Compute masses
    double m_fuel = 1.0 * MW_array[iFUEL];
    double m_O2   = X_O2 * mw_o2;
    double m_N2   = X_O2 * AIR_N2_O2_RATIO * mw_n2;
    double m_tot  = m_fuel + m_O2 + m_N2;
    
    // Assign local mass fractions
    Y_res[iFUEL] = m_fuel / m_tot;
    Y_res[iO2]   = m_O2 / m_tot;
    Y_res[iN2]   = m_N2 / m_tot;
}


// ===================================================================
// --- Boundary Conditions ---
// ===================================================================

// Bottom: Injection
#if BG_CONFIG == 0 || BG_CONFIG == 3
  u.n[bottom] = dirichlet( 0. ); 
#else
  u.n[bottom] = dirichlet( V_INJ ); 
#endif
u.t[bottom] = dirichlet( 0. );     
p[bottom]   = neumann( 0. );       
pf[bottom]  = neumann( 0. );

// Top: Sortie (Outflow)
u.n[top]    = neumann( 0. );       
u.t[top]    = neumann( 0. );       
p[top]      = dirichlet( 0. );   
pf[top]     = dirichlet( 0. );

// Left & Right
#if WALL_CONFIG == 0
  // Wall
  u.n[left]   = dirichlet( 0. );     
  u.t[left]   = dirichlet( 0. );    
  p[left]     = neumann( 0. );       
  pf[left]    = neumann( 0. );
  u.n[right]  = dirichlet( 0. );     
  u.t[right]  = dirichlet( 0. );     
  p[right]    = neumann( 0. );       
  pf[right]   = neumann( 0. );
#else
  // Outflow
  u.n[left]   = neumann( 0. );     
  u.t[left]   = neumann( 0. );    
  p[left]     = dirichlet( 0. );       
  pf[left]    = dirichlet( 0. );

  u.n[right]  = neumann( 0. );     
  u.t[right]  = neumann( 0. );     
  p[right]    = dirichlet( 0. );       
  pf[right]   = dirichlet( 0. );
#endif

// --- Computation Tool ---
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
    boundary (YList);
}

// ===================================================================
// --- 1D flame import and interpolation ---
// ===================================================================

typedef struct {
    int n_points;
    double *x;
    double *T;
    double *u;
    double **Y; 
} FlameProfile;

FlameProfile flame_prof;

double interpolate_1D(double x_target, double * x_array, double * y_array, int n) {
    if (n == 0) return 0.0;
    if (x_target <= x_array[0]) return y_array[0];
    if (x_target >= x_array[n-1]) return y_array[n-1];
    
    int i = 0;
    while (i < n - 1 && x_array[i+1] < x_target) i++;
    
    double dx = x_array[i+1] - x_array[i];
    if (dx < 1e-12) return y_array[i]; 
    
    double t = (x_target - x_array[i]) / dx;
    return y_array[i] + t * (y_array[i+1] - y_array[i]);
}

void load_flame_csv(const char * filename) {
    FILE * fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Fatal Error : Can not read this csv %s\n", filename);
        exit(1);
    }
    
    int lines = 0;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), fp)) lines++;
    flame_prof.n_points = lines - 1; 
    
    flame_prof.x = (double *)malloc(flame_prof.n_points * sizeof(double));
    flame_prof.T = (double *)malloc(flame_prof.n_points * sizeof(double));
    flame_prof.u = (double *)malloc(flame_prof.n_points * sizeof(double));
    flame_prof.Y = (double **)malloc(NS * sizeof(double *));
    
    for (int s = 0; s < NS; s++) {
        flame_prof.Y[s] = (double *)malloc(flame_prof.n_points * sizeof(double));
        for (int i = 0; i < flame_prof.n_points; i++) flame_prof.Y[s][i] = 0.0;
    }
    
    rewind(fp);
    fgets(buffer, sizeof(buffer), fp);
    
    int col_mapping[500]; 
    for(int i=0; i<500; i++) col_mapping[i] = -1;
    
    int col_idx = 0;
    char * token = strtok(buffer, ", \n\r");
    while (token != NULL) {
        if (strcmp(token, "x") == 0) col_mapping[col_idx] = -3;
        else if (strcmp(token, "T") == 0) col_mapping[col_idx] = -2;
        else if (strncmp(token, "Y_", 2) == 0) {
            int s_idx = index_species(token + 2); 
            if (s_idx != -1) col_mapping[col_idx] = s_idx;
        }
        col_idx++;
        token = strtok(NULL, ", \n\r");
    }
    
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
        printf("INFO : Flame profile successfully loaded (%d points).\n", flame_prof.n_points);
    }
}

// ===================================================================
// --- Main ---
// ===================================================================

int main (int argc, char ** argv) {

    kinetics(KIN_MECHANISM, &NS);
    if (NS <= 0) return 1;

    // --- Atomic Composition Initialization ---
    atoms_C = (double *)malloc(NS * sizeof(double));
    atoms_H = (double *)malloc(NS * sizeof(double));
    atoms_O = (double *)malloc(NS * sizeof(double));
    MW_array = (double *)malloc(NS * sizeof(double));

    molecular_weights(NS, MW_array); 

    // Retrieve Cantera element indices 
    size_t idx_C = thermo_elementIndex(thermo, "C");
    size_t idx_H = thermo_elementIndex(thermo, "H");
    size_t idx_O = thermo_elementIndex(thermo, "O");
    size_t num_elements = thermo_nElements(thermo);

    for (int k = 0; k < NS; k++) {
        // Retrieve number of atoms per species, safely checking if element exists in the mechanism
        atoms_C[k] = (idx_C < num_elements) ? thermo_nAtoms(thermo, k, idx_C) : 0.0;
        atoms_H[k] = (idx_H < num_elements) ? thermo_nAtoms(thermo, k, idx_H) : 0.0;
        atoms_O[k] = (idx_O < num_elements) ? thermo_nAtoms(thermo, k, idx_O) : 0.0;
    }

    // Retrieve indices for key species using the Cantera wrapper
    int iFUEL = index_species(FUEL_NAME);
    int iO2   = index_species("O2");
    int iN2   = index_species("N2");

    if (BG_CONFIG == 3) {
        double Y_bottom_temp[NS];
        get_Y_from_atomic_phi(PHI_BOTTOM, iFUEL, iO2, iN2, Y_bottom_temp);
        Y_FUEL_B = Y_bottom_temp[iFUEL];
        Y_O2_B   = Y_bottom_temp[iO2];
        Y_N2_B   = Y_bottom_temp[iN2];
    }

    if (iFUEL >= 0 && iO2 >= 0 && iN2 >= 0) {
        // Retrieve atomic counts for the fuel 
        double nC = atoms_C[iFUEL];
        double nH = atoms_H[iFUEL];
        double nO = atoms_O[iFUEL];
        
        // Calculate theoretical stoichiometric O2 moles required for 1 mole of fuel
        double stoich_O2_moles = (4.0 * nC + nH - 2.0 * nO) / 4.0;

        double m_fuel = 1.0 * MW_array[iFUEL];
        double m_O2   = stoich_O2_moles * MW_array[iO2];
        double m_N2   = stoich_O2_moles * AIR_N2_O2_RATIO * MW_array[iN2];
        double m_tot_stoich = m_fuel + m_O2 + m_N2;
        double m_air        = m_O2 + m_N2;

        Y_FUEL_STOICH = m_fuel / m_tot_stoich;
        Y_O2_STOICH   = m_O2   / m_tot_stoich;
        Y_N2_STOICH   = m_N2   / m_tot_stoich;
        Y_O2_AIR      = m_O2 / m_air;
        Y_N2_AIR      = m_N2 / m_air;
        
        if (pid() == 0) {
            printf("INFO: Dynamic stoichiometry computed successfully.\n");
            printf("INFO: Y_FUEL_STOICH = %.6f, Y_O2_STOICH = %.6f\n", Y_FUEL_STOICH, Y_O2_STOICH);
        }
    } else {
        if (pid() == 0) {
            fprintf(stderr, "FATAL ERROR: Could not find FUEL, O2, or N2 in the mechanism.\n");
        }
        exit(1);
    }

    gas_species = new_species_names(NS);

    load_flame_csv("flame_profile_isoprop.csv"); 
    
    origin(0., 0.);
    size(DOMAIN_SIZE);

    G = (coord){0., -9.81, 0.}; 
    CFL = CFL_MAX; 
    Pref = P_ATM;
    T0   = T_INITIAL;

    NITERMAX = 150;                 // Maximum number of iterations
    TOLERANCE = 1e-2;               // Convergence tolerances

    init_grid(1 << MIN_LEVEL);
    run();

    // Clean up dynamic arrays
    if (flame_prof.x != NULL) free(flame_prof.x);
    if (flame_prof.T != NULL) free(flame_prof.T);
    if (flame_prof.u != NULL) free(flame_prof.u);
    if (flame_prof.Y != NULL) {
        for(int s = 0; s < NS; s++) if (flame_prof.Y[s] != NULL) free(flame_prof.Y[s]);
        free(flame_prof.Y);
    }
    free_species_names(NS, gas_species);

    return 0;
}

// ===================================================================
// --- Initialization Event ---
// ===================================================================

event init_0 (i = 0) {
  // Access Basilisk scalar fields strictly through the multiphase gas structure[cite: 2]
  scalar T_gas = gas->T;
  scalar * YList = gas->YList;

  // Extract species indices using the Cantera C-interface wrapper[cite: 3]
  int iFUEL = index_species(FUEL_NAME);
  int iO2   = index_species("O2");
  int iN2   = index_species("N2");

  // Define geometric parameters for the physical ignition kernels
  double R_core = 0.4 * R_FLAME;                  
  double flame_thickness_0 = R_FLAME - R_core;      
  
  // Specific geometric constraints for the vertical linear flame (Config 1)
  double X_core = 0.4 * X_FLAME; 
  double flame_thickness_1 = X_FLAME - X_core;
  
  // End of the 1D profile corresponding to the fully burned thermodynamic state
  double x_max_csv = flame_prof.x[flame_prof.n_points - 1]; 

  foreach() { 
      double T_loc = T_INITIAL;
      double Y_loc[NS];
      
      // Establish background composition (stratified or homogeneous fresh mixture)
      if (BG_CONFIG == 3) {
          double phi_local_bg;
          if (y <= Y_STRAT) {
              double y_star = y / Y_STRAT;
              double delta_phi = PHI_TOP - PHI_BOTTOM;
              
              if (STRAT_PROFILE == 0) { 
                  // Linear distribution
                  phi_local_bg = PHI_BOTTOM + delta_phi * y_star;
              } 
              else if (STRAT_PROFILE == 1) { 
                  // Exponential distribution
                  phi_local_bg = PHI_BOTTOM + delta_phi * 
                                 (exp(STRAT_CURVATURE * y_star) - 1.0) / 
                                 (exp(STRAT_CURVATURE) - 1.0);
              } 
              else if (STRAT_PROFILE == 2) { 
                  // Logarithmic distribution
                  phi_local_bg = PHI_BOTTOM + delta_phi * 
                                 log(1.0 + STRAT_CURVATURE * y_star) / 
                                 log(1.0 + STRAT_CURVATURE);
              }
          } else {
              phi_local_bg = PHI_TOP;
          }
          
          // Physical clipping to prevent non-physical equivalence ratios
          phi_local_bg = max(0.0, phi_local_bg);
          get_Y_from_atomic_phi(phi_local_bg, iFUEL, iO2, iN2, Y_loc);
          
      } else {
          for (int s = 0; s < NS; s++) Y_loc[s] = 0.0;
          if (iFUEL != -1) Y_loc[iFUEL] = Y_FUEL_STOICH;
          if (iO2 != -1)   Y_loc[iO2]   = Y_O2_STOICH;
          if (iN2 != -1)   Y_loc[iN2]   = Y_N2_STOICH;
      }
      
      bool in_flame = false;
      double x_csv = 0.0;

      // ---------------------------------------------------------
      // CONFIG 0: Spherical Flame
      // ---------------------------------------------------------
      if (IGN_CONFIG == 0) {
          double r = sqrt(sq(x - X_FLAME) + sq(y - Y_flame));
          if (r <= R_FLAME && x >= 0.0) {
              if (r <= R_core) {
                  x_csv = x_max_csv; 
              } else {
                  x_csv = (r - R_core) / flame_thickness_0 * x_max_csv; 
              }
              in_flame = true;
          }
      } 
      // ---------------------------------------------------------
      // CONFIG 1: Vertical Linear Flame (Wall) with Tip Smoothing
      // ---------------------------------------------------------
      else if (IGN_CONFIG == 1 && x >= 0.0) {
          if (y <= H_FLAME) {
              // Main body of the vertical flame
              if (x <= X_core) {
                  x_csv = x_max_csv; // Burned gas core enforcing zero gradient
                  in_flame = true;
              } else if (x <= X_FLAME) {
                  double r_side = x - X_core;
                  x_csv = (flame_thickness_1 - r_side) / flame_thickness_1 * x_max_csv;
                  in_flame = true;
              }
          } else {
              // Smooth radial closure at the tip of the flame to avoid infinite gradients
              double dy = y - H_FLAME;
              double dx = (x > X_core) ? (x - X_core) : 0.0;
              double r_tip = sqrt(sq(dx) + sq(dy));
              
              if (r_tip <= flame_thickness_1) {
                  x_csv = (flame_thickness_1 - r_tip) / flame_thickness_1 * x_max_csv;
                  in_flame = true;
              }
          }
      }
      // ---------------------------------------------------------
      // CONFIG 2: Horizontal Linear Flame
      // ---------------------------------------------------------
      else if (IGN_CONFIG == 2 && y >= Y_flame) {
          in_flame = true; 
          x_csv = y - Y_flame; 
      }

      // Interpolate thermodynamic state and mass fractions from the 1D profile
      if (in_flame) {
          T_loc = interpolate_1D(x_csv, flame_prof.x, flame_prof.T, flame_prof.n_points);
          for (int s = 0; s < NS; s++) {
              Y_loc[s] = interpolate_1D(x_csv, flame_prof.x, flame_prof.Y[s], flame_prof.n_points);
          }
      }
      
      // Assign local computations to the Basilisk phase scalar fields[cite: 2]
      T_gas[] = T_loc;
      for (int s = 0; s < NS; s++) {
          scalar Y = YList[s];
          Y[] = Y_loc[s];
      }
      
      u.x[] = 0.; 
      u.y[] = 0.; 
      p[]   = 0.;
  }

  boundary((scalar *){u.x, u.y, p});

  // Apply gradient limiters for scalars to avoid spurious numerical oscillations
  for (scalar s in YList) {
      s.gradient = minmod2;
  }
  T_gas.gradient = minmod2;

  // Apply Boundary Conditions using strictly global variables to avoid qcc extraction errors
  if (BG_CONFIG == 1 || BG_CONFIG == 2 || BG_CONFIG == 3) {
      T_gas[bottom] = dirichlet(300.0); 

      for (int s = 0; s < NS; s++) {
          scalar Y = YList[s];
          if (BG_CONFIG == 2) { 
              if (s == iFUEL) { Y[bottom] = dirichlet(1.0); } 
              else { Y[bottom] = dirichlet(0.0); } 
          } 
          else if (BG_CONFIG == 1) { 
              if (s == iFUEL) { Y[bottom] = dirichlet(Y_FUEL_STOICH); } 
              else if (s == iO2) { Y[bottom] = dirichlet(Y_O2_STOICH); } 
              else if (s == iN2) { Y[bottom] = dirichlet(Y_N2_STOICH); } 
              else { Y[bottom] = dirichlet(0.0); } 
          }
          else if (BG_CONFIG == 3) { 
              if (s == iFUEL) { Y[bottom] = dirichlet(Y_FUEL_B); } 
              else if (s == iO2) { Y[bottom] = dirichlet(Y_O2_B); } 
              else if (s == iN2) { Y[bottom] = dirichlet(Y_N2_B); } 
              else { Y[bottom] = dirichlet(0.0); } 
          }
      } 
  }

  boundary({T_gas}); 
  boundary(YList); 
  
  // Ensure mass fractions sum strictly to 1.0 safely
  sanitize_fractions(YList); 
  
  // Trigger calculation of density via cantera_gasprop_density to satisfy the ideal gas law[cite: 5]
  event("properties"); 
  
  // Safely initialize the density variation source terms to zero
  // Prevents non-physical expansion spikes in the Low-Mach projection step[cite: 6]
  foreach() {
      for (scalar drhodt_s in drhodtlist) {
          drhodt_s[] = 0.;
      }
      for (scalar intexp_s in intexplist) {
          intexp_s[] = 0.;
      }
  }

  // Protect I/O operations for HPC/MPI compatibility
  if (pid() == 0) {
      printf("INFO: Physical flame kernel and stratified background initialized successfully.\n");
  }
}

// ===================================================================
// --- Residency of the flame ---
// ===================================================================

event flame_residence (t <= T_RESIDENCE) {
  // Access phase scalar fields[cite: 3]
  scalar T_gas = gas->T;
  scalar * YList = gas->YList;

  // Consistency with the initial physical kernel geometry
  double R_core = 0.4 * R_FLAME;                  
  double flame_thickness_0 = R_FLAME - R_core;      
  
  double X_core = 0.4 * X_FLAME; 
  double flame_thickness_1 = X_FLAME - X_core;
  
  double x_max_csv = flame_prof.x[flame_prof.n_points - 1]; 

  // Maintain the mapped flame profile dynamically
  foreach() {
      bool in_flame = false;
      double x_csv = 0.0;

      // CONFIG 0: Spherical Flame
      if (IGN_CONFIG == 0) {
          double r = sqrt(sq(x - X_FLAME) + sq(y - Y_flame));
          if (r <= R_FLAME && x >= 0.0) {
              if (r <= R_core) { x_csv = x_max_csv; } 
              else { x_csv = (r - R_core) / flame_thickness_0 * x_max_csv; }
              in_flame = true;
          }
      } 
      // CONFIG 1: Vertical Linear Flame (Wall) with Tip Smoothing
      else if (IGN_CONFIG == 1 && x >= 0.0) {
          if (y <= H_FLAME) {
              if (x <= X_core) {
                  x_csv = x_max_csv; 
                  in_flame = true;
              } else if (x <= X_FLAME) {
                  double r_side = x - X_core;
                  x_csv = (flame_thickness_1 - r_side) / flame_thickness_1 * x_max_csv;
                  in_flame = true;
              }
          } else {
              double dy = y - H_FLAME;
              double dx = (x > X_core) ? (x - X_core) : 0.0;
              double r_tip = sqrt(sq(dx) + sq(dy));
              
              if (r_tip <= flame_thickness_1) {
                  x_csv = (flame_thickness_1 - r_tip) / flame_thickness_1 * x_max_csv;
                  in_flame = true;
              }
          }
      }
      // CONFIG 2: Horizontal Linear Flame
      else if (IGN_CONFIG == 2 && y >= Y_flame) {
          in_flame = true; 
          x_csv = y - Y_flame; 
      }

      if (in_flame) {
          T_gas[] = interpolate_1D(x_csv, flame_prof.x, flame_prof.T, flame_prof.n_points);
          for (int s = 0; s < NS; s++) {
              scalar Y = YList[s];
              Y[] = interpolate_1D(x_csv, flame_prof.x, flame_prof.Y[s], flame_prof.n_points);
          }
      }
  }

  boundary({T_gas}); 
  boundary(YList); 
  
  sanitize_fractions(YList);

  // Trigger thermodynamic properties update 
  event("properties"); 

  // Reset Low-Mach divergence sources to avoid spurious expansion[cite: 7]
  foreach() {
      for (scalar drhodt_s in drhodtlist) {
          drhodt_s[] = 0.;
      }
      for (scalar intexp_s in intexplist) {
          intexp_s[] = 0.;
      }
  }
}

// =================================================================
// --- AMR ---
// =================================================================

event adapt (i++) {
    scalar * list = NULL;
    
    list = list_append(list, gas->T);
    list = list_append(list, u.x);
    list = list_append(list, u.y);
    
    int iFUEL = index_species(FUEL_NAME);
    int iOH   = index_species("OH"); 
    int iCO   = index_species("CO"); 
    
    if (iFUEL >= 0) list = list_append(list, gas->YList[iFUEL]);
    if (iOH >= 0)   list = list_append(list, gas->YList[iOH]);
    if (iCO >= 0)   list = list_append(list, gas->YList[iCO]);
    
    int num_scalars = list_len(list);
    double thresholds[num_scalars];
    
    int idx = 0;
    thresholds[idx++] = 5.0;   
    thresholds[idx++] = 0.02;  
    thresholds[idx++] = 0.02;  
    
    if (iFUEL >= 0) thresholds[idx++] = 1e-3; 
    if (iOH >= 0)   thresholds[idx++] = 1e-5; 
    if (iCO >= 0)   thresholds[idx++] = 1e-4; 
    
    adapt_wavelet (list, thresholds, maxlevel = MAX_LEVEL, minlevel = MIN_LEVEL);
    free (list);
}

// =================================================================
// --- Computations ---
// =================================================================

// Heat Release Rate
event compute_hrr (i++) {
    int ns = NS; 
    scalar * YList = gas->YList;
    scalar T_gas = gas->T;
    
    int idx_FUEL = index_species(FUEL_NAME); 
    
    foreach() {
        double ymass[ns], hm[ns], wdot[ns];
        for (int s = 0; s < ns; s++) {
            scalar Y = YList[s];
            ymass[s] = Y[]; 
        }
        
        thermo_setTemperature(thermo, T_gas[]);
        thermo_setPressure(thermo, P_ATM); 
        thermo_setMassFractions(thermo, ns, ymass, 1);
        
        thermo_getPartialMolarEnthalpies(thermo, ns, hm); 
        kin_getNetProductionRates(kin, ns, wdot);
        
        double hrr_local = 0.0;
        for (int s = 0; s < ns; s++) hrr_local -= wdot[s] * hm[s];
        HRR[] = hrr_local;
        
        // Fix: Use the dynamic molecular weight array instead of the deleted macro
        if (idx_FUEL >= 0) wdot_fuel[] = wdot[idx_FUEL] * MW_array[idx_FUEL];
        else wdot_fuel[] = 0.0;
    }
    boundary({HRR, wdot_fuel}); 
}

// Global flame speed (with csv export)
event monitor_speed (i += 10) { 
    double sum_wdot = 0.0; 
    double A_front = 0.0; 
    
    boundary({gas->T}); 

    foreach(reduction(+:sum_wdot) reduction(+:A_front)) {
        sum_wdot += wdot_fuel[] * dv();
        
        double delta_T = 2220.0 - 300.0; // T_ad approximative
        double grad_Tx = (gas->T[1,0] - gas->T[-1,0]) / (2.0 * Delta);
        double grad_Ty = (gas->T[0,1] - gas->T[0,-1]) / (2.0 * Delta);
        double grad_c = sqrt(sq(grad_Tx) + sq(grad_Ty)) / delta_T;
        
        A_front += grad_c * dv();
    }

    if (pid() == 0) {
        double rho_0 = 1.16;       
        double Sc = 0.0;
        
        if (A_front > 1e-6) Sc = -sum_wdot / (rho_0 * Y_FUEL_STOICH * A_front); 
        
        static FILE * fp = fopen("flame_speed.csv", "w");
        if (i == 0) fprintf(fp, "time,area_m,speed_cm_s\n");
        fprintf(fp, "%g,%g,%g\n", t, A_front, Sc * 100.0);
        fflush(fp);
    }
}

// Local flame speed (not working for now)
event compute_local_speed (i++) {
    double rho_0 = 1.16;
    double delta_T = 2220.0 - 300.0;

    boundary({gas->T, wdot_fuel, HRR});

    foreach() {
        double grad_Tx = (gas->T[1,0] - gas->T[-1,0]) / (2.0 * Delta);
        double grad_Ty = (gas->T[0,1] - gas->T[0,-1]) / (2.0 * Delta);
        double grad_c = sqrt(sq(grad_Tx) + sq(grad_Ty)) / delta_T;

        if (HRR[] > 1e5) { 
            if (grad_c > 1000.0) S_local[] = 100.0 * (-wdot_fuel[]) / (rho_0 * Y_FUEL_STOICH * grad_c);
            else S_local[] = 0.0;
        } else {
            S_local[] = 0.0;
        }
    }
    boundary({S_local});
}

// Computing local phi based on Bilger mixture fraction
event compute_phi (i++) {
    scalar * YList = gas->YList; // Retrieve mass fractions
    
    foreach() {
        double num = 0.0;
        double den = 0.0;
        
        for (int k = 0; k < NS; k++) {
            scalar Y = YList[k];
            // Compute local moles (proportional to mole fraction X_k)
            double moles_k = Y[] / MW_array[k]; 
            
            num += moles_k * (4.0 * atoms_C[k] + atoms_H[k]);
            den += moles_k * (2.0 * atoms_O[k]);
        }
        
        // Prevent division by zero in pure fuel or inert regions
        if (den > 1e-12) {
            phi_local[] = num / den;
        } else {
            // Apply artificial cap to avoid singularities in the solver
            phi_local[] = (num > 1e-12) ? 100.0 : 0.0; 
        }
    }
    boundary({phi_local});
}

// ===================================================================
// --- Monitoring Logs (Original Implementation) ---
// ===================================================================

event logfile (i += 10) { 
    double T_max = -HUGE, T_min = HUGE;
    double P_max = -HUGE, P_min = HUGE;
    double div_max = -HUGE, div_min = HUGE;
    double sum_Y = 0.0;
    
    foreach(reduction(max:T_max) reduction(min:T_min) 
            reduction(max:P_max) reduction(min:P_min)
            reduction(max:div_max) reduction(min:div_min) 
            reduction(+:sum_Y)) {
        
        T_max = max(T_max, gas->T[]); 
        T_min = min(T_min, gas->T[]);
        P_max = max(P_max, p[]); 
        P_min = min(P_min, p[]);
        double div = 0.0;

        foreach_dimension() {
            div += (u.x[1] - u.x[-1]) / (2.0 * Delta);
        }
        div_max = max(div_max, div); 
        div_min = min(div_min, div);
        
        for (scalar s in gas->YList) {
            sum_Y += s[];
        }
    }

    if (pid() == 0) {
        time_t rawtime; 
        struct tm * timeinfo; 
        char time_buffer[80];
        time(&rawtime); 
        timeinfo = localtime(&rawtime);
        strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        
        double avg_Y = sum_Y / (double)grid->tn; 
        
        // Truncate and initialize file at i = 0, then append for subsequent iterations
        FILE * fp = fopen("log.txt", (i == 0) ? "w" : "a");
        if (fp != NULL) {
            fprintf(fp, 
                "[%s] i: %-5d | t: %-8.4g | dt: %-8.4g | Cells(Tot/Loc): %ld / %ld | T: %.0f/%.0f K | P_dyn: %.2f/%.2f Pa | Div: %e | AvgY: %.4f\n", 
                time_buffer, i, t, dt, (long)grid->tn, (long)grid->n, T_min, T_max, P_min, P_max, div_max, avg_Y);
            fclose(fp);
        } else {
            fprintf(ferr, "Warning: Could not open detailed_log.txt for writing.\n");
        }
    }
}

// =================================================================
// --- VTK EXPORT ---
// =================================================================

void sanitize_vtk_name(const char * input, char * output) {
    int i = 0;
    while(input[i] != '\0' && i < 255) {
        if(input[i] == '(' || input[i] == ')' || input[i] == '-' || input[i] == '+') output[i] = '_';
        else output[i] = input[i];
        i++;
    }
    output[i] = '\0';
}

event snapshot_vtu (t += DT; t <= T_END) {
    if (pid() == 0) { mkdir("vtk_pieces", 0777); mkdir("vtu", 0777); }
    MPI_Barrier(MPI_COMM_WORLD); 

    int ncells = 0; foreach(serial) ncells++;
    int npoints = ncells * 4; 

    char name[256]; sprintf(name, "vtk_pieces/fields_t_%.4f_n%d.vtu", t, pid());
    FILE * fp = fopen(name, "w");
    
    // Wrap the rest of the export logic inside an if block to avoid premature return statements
    if (fp) {
        fprintf(fp, "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
        fprintf(fp, "  <UnstructuredGrid>\n    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", npoints, ncells);

        fprintf(fp, "      <Points>\n        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        foreach() {
            double d = Delta / 2.0;
            fprintf(fp, "%.8g %.8g 0.0\n%.8g %.8g 0.0\n%.8g %.8g 0.0\n%.8g %.8g 0.0\n", x-d, y-d, x+d, y-d, x+d, y+d, x-d, y+d); 
        }
        fprintf(fp, "        </DataArray>\n      </Points>\n");

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

        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"wdot_fuel\" format=\"ascii\">\n");
        foreach() fprintf(fp, "%.8g\n", wdot_fuel[]);
        fprintf(fp, "        </DataArray>\n"); 

        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"S_local\" format=\"ascii\">\n");
        foreach() fprintf(fp, "%.8g\n", S_local[]);
        fprintf(fp, "        </DataArray>\n");

        fprintf(fp, "        <DataArray type=\"Float64\" Name=\"phi_local\" format=\"ascii\">\n");
        foreach() fprintf(fp, "%.8g\n", phi_local[]);
        fprintf(fp, "        </DataArray>\n");

        for (int s = 0; s < NS; s++) {
            char safe_name[256]; sanitize_vtk_name(gas_species[s], safe_name);
            fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", safe_name);
            scalar Y = gas->YList[s];
            foreach() fprintf(fp, "%.8g\n", isfinite(Y[]) ? Y[] : 0.0);
            fprintf(fp, "        </DataArray>\n");
        }
        fprintf(fp, "      </CellData>\n"); 

        fprintf(fp, "      <Cells>\n        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
        for (int i = 0; i < ncells; i++) fprintf(fp, "%d %d %d %d\n", i*4, i*4+1, i*4+2, i*4+3); 
        fprintf(fp, "        </DataArray>\n        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
        for (int i = 0; i < ncells; i++) fprintf(fp, "%d\n", (i+1)*4); 
        fprintf(fp, "        </DataArray>\n        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
        for (int i = 0; i < ncells; i++) fprintf(fp, "9\n"); 
        fprintf(fp, "        </DataArray>\n      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
        fclose(fp);
    }

    if (pid() == 0) {
        char master_name[256]; sprintf(master_name, "vtu/fields_t_%.4f.pvtu", t);
        FILE * fpvtu = fopen(master_name, "w");
        if (fpvtu) {
            fprintf(fpvtu, "<?xml version=\"1.0\"?>\n<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
            fprintf(fpvtu, "  <PUnstructuredGrid GhostLevel=\"0\">\n    <PPoints>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\"/>\n    </PPoints>\n");
            fprintf(fpvtu, "    <PCellData>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"T\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"HRR\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"wdot_fuel\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"S_local\" format=\"ascii\"/>\n");
            fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"phi_local\" format=\"ascii\"/>\n");

            for (int s = 0; s < NS; s++) {
                char safe_name[256]; sanitize_vtk_name(gas_species[s], safe_name);
                fprintf(fpvtu, "      <PDataArray type=\"Float64\" Name=\"%s\" format=\"ascii\"/>\n", safe_name);
            }
            fprintf(fpvtu, "    </PCellData>\n");
            for (int i = 0; i < npe(); i++) fprintf(fpvtu, "    <Piece Source=\"../vtk_pieces/fields_t_%.4f_n%d.vtu\"/>\n", t, i);
            fprintf(fpvtu, "  </PUnstructuredGrid>\n</VTKFile>\n");
            fclose(fpvtu);
        }
    }
}


// =================================================================
// --- Video Output ---
// =================================================================

event movie (t += DT; t <= T_END) {         
  clear(); 
  view (tx = -0.5, ty = -0.5);
  squares ("T", min = 300, max = 2500, linear = true); 
  save ("temperature_evolution.mp4"); 
}

event cleanup_properties (t = end) {
    if (atoms_C != NULL) free(atoms_C);
    if (atoms_H != NULL) free(atoms_H);
    if (atoms_O != NULL) free(atoms_O);
    if (MW_array != NULL) free(MW_array);
}