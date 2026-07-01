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
#define MAX_LEVEL       12            // Maximum grid refinement level
#define MIN_LEVEL       8            // Minimum grid refinement level

// --- Simulation Constants ---
#define T_END           0.05         // Final time (seconds)
#define CFL_MAX         0.1         // Stability criterion
#define P_ATM           101325.0     // Atmospheric pressure (Pa)
#define T_INITIAL       300.0        // Initial gas temperature (K)

// --- Simulation Parameters ---
#define V_INJ 0.1                  // Injection speed (m/s) 
#define F_FUEL 1                     // Fuel Ratio
// --- Chemistry Mechanism ---
// Note: Ensure the path is relative to the execution directory 
// or define the full absolute path.
#define KIN_MECHANISM   "laminarflame.yaml"

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

// ----------------------------------------------
// --- Main ---
// ----------------------------------------------

int main (int argc, char ** argv) {

    fprintf(stderr, "DEBUG: Address of NS in main: %p\n", (void*)&NS);
    kinetics(KIN_MECHANISM, &NS);
    if (NS <= 0) {
        fprintf(stderr, "FATAL: NS is %d. Kinetics failed.\n", NS);
        return 1;
    }

    gas_species = new_species_names(NS);
    fprintf(stderr, "DEBUG: Species allocated, NS=%d\n", NS);

    origin(0., 0.);
    size(DOMAIN_SIZE);
    
    CFL = 0.2; 
    Pref = P_ATM;
    T0   = T_INITIAL;
    DT = 1e-9;

    NITERMAX = 100;
    TOLERANCE = 1e-1;

    init_grid(1 << MIN_LEVEL);

    run();

    free_species_names(NS, gas_species);
    return 0;
}


// ----------------------------------------------
// --- Initialisation ---
// ----------------------------------------------
event init (i = 0) {
  // 1. Initialisation de base (Vitesse et Pression Dynamique)
  foreach() { 
      u.x[] = 0.; 
      u.y[] = V_INJ; 
      p[]   = 0.;
  }
  boundary((scalar *){u.x, u.y, p});

  // 2. Initialisation thermodynamique Cantera GLOBALE
  int iH2  = index_species ("H2");
  int iO2  = index_species ("O2");
  int iN2  = index_species ("N2");
  int iH2O = index_species ("H2O");

  double x[NS];
  for (int s = 0; s < NS; s++) x[s] = 0.0;
  if (iO2 != -1 && iN2 != -1) { x[iO2] = 0.233; x[iN2] = 0.767; }

  ThermoState tsg = {T0, Pref, x};
  phase_set_thermo_state (gas, &tsg, force = !restore (file = "restart"));

  // 3. Sécurisation des gradients
  for (scalar s in gas->YList) s.gradient = minmod2;
  gas->T.gradient = minmod2;

  // 4. Noyau d'allumage
  
  // ---> CORRECTION A : On fait la remise à zéro de TOUTES les espèces 
  // en mettant le foreach() à l'INTÉRIEUR du for() (Basilisk adore ça).
  for (scalar s in gas->YList) {
      foreach() s[] = 0.0;
  }

  // ---> CORRECTION B : Extraction de scalaires simples et sécurisés
  // Si l'espèce n'existe pas (-1), on crée un scalaire "vide" pour éviter un segfault
  scalar T_loc = gas->T;
  scalar fuel  = (iH2 != -1)  ? gas->YList[iH2]  : (scalar){-1};
  scalar oxi   = (iO2 != -1)  ? gas->YList[iO2]  : (scalar){-1};
  scalar inert = (iN2 != -1)  ? gas->YList[iN2]  : (scalar){-1};
  scalar h2o   = (iH2O != -1) ? gas->YList[iH2O] : (scalar){-1};

  double xc = 3e-3, yc = 3e-3, r_ign = 0.002, T_ign = 1200.0;

  // ---> CORRECTION C : Le foreach ne contient plus aucune liste, 
  // seulement des scalaires locaux.
  foreach() {
      double r = sqrt(sq(x - xc) + sq(y - yc));
      double f = 0.5 * (1.0 - tanh((r - r_ign)/0.0002)); 
  
      T_loc[] = T0 * (1.0 - f) + T_ign * f;
  
      // On utilise la propriété ".i" du scalaire pour vérifier qu'il est valide
      if (fuel.i >= 0)  fuel[]  = 0.0;
      if (oxi.i >= 0)   oxi[]   = 0.233 * (1.0 - f);
      if (h2o.i >= 0)   h2o[]   = 0.25 * f;
      if (inert.i >= 0) inert[] = 0.767 * (1.0 - f) + 0.75 * f;
  }

  // 5. Clamping Absolu de la température
  foreach() {
    if (T_loc[] < 300.0) T_loc[] = 300.0; 
  }
  boundary({T_loc});
  
  // 6. NETTOYAGE THERMO OBLIGATOIRE
  sanitize_fractions(gas->YList); 

  // 7. Calcul des propriétés
  event("properties"); 

  // 8. Calcul initial de alpha 
  foreach() {
    if (rho[] < 0.01) rho[] = 1.0; 
  }
  foreach_face() {
      double rhof = (rho[] + rho[-1]) / 2.0;
      alpha.x[] = 1.0 / rhof; 
  }
  boundary ((scalar *){alpha.x, alpha.y});
}


event adapt (i++) {
  scalar * list = NULL;
  list = list_append(list, gas->T); 
  for (scalar s in gas->YList) {
    list = list_append(list, s);
  }
  
  // Création d'un tableau de seuils de la bonne taille exacte
  int num_scalars = list_len(list);
  double * thresholds = malloc(num_scalars * sizeof(double));
  
  // Remplissage : 0.05 pour T (le premier), 0.01 pour les autres
  thresholds[0] = 0.05; 
  for (int j = 1; j < num_scalars; j++) {
      thresholds[j] = 0.01;
  }
  
  // Adaptation sécurisée
  adapt_wavelet (list, thresholds, maxlevel = MAX_LEVEL, minlevel = MIN_LEVEL);
                 
  // Libération totale de la mémoire
  free (thresholds);
  free (list);
}


// ----------------------------------------------
// --- DEBUG ---
// ----------------------------------------------

event monitoring (i++) {
    // 1. Calcul des stats globales pour T et P
    stats sT = statsf (gas->T);
    stats sp = statsf (p);
    
    // 2. Calcul de la somme des espèces
    scalar Y_sum[]; 
    foreach() {
        double sum = 0.;
        for (scalar s in gas->YList) sum += s[];
        Y_sum[] = sum;
    }
    stats sY = statsf (Y_sum);
    
    // 3. Comptage des cellules locales via REDUCTION
    long local_cells = 0;
    foreach(reduction(+:local_cells)) {
        local_cells++;
    }
    
    // 4. Affichage (Ajout de t et dt)
    // t : temps courant, dt : pas de temps pour l'itération suivante
    fprintf(stderr, "MONITORING i=%d [PID %d] | t=%g dt=%g | Cells: %ld | T:[%g, %g] | P:[%g, %g] | Y_sum:[%g, %g]\n", 
            i, pid(), t, dt, local_cells, sT.min, sT.max, sp.min, sp.max, sY.min, sY.max);
    fflush(stderr);

    // 5. Arrêt de sécurité
    /*if (isnan(sT.max) || sT.max > 5000.0 || sY.max > 1.1 || sY.min < 0.9) {
        fprintf(stderr, "FATAL: Divergence à i=%d. Arrêt.\n", i);
        abort(); 
    }*/
}

event diagnostic (i++) {
    // 1. Calcul des statistiques globales pour T et p (géré par Basilisk)
    stats sT = statsf (gas->T);
    stats sp = statsf (p);
    
    // 2. Calcul de la somme des espèces par cellule
    // On crée un scalaire temporaire pour stocker la somme locale
    scalar Y_sum[]; 
    foreach() {
        double sum = 0.;
        for (scalar s in gas->YList) {
            sum += s[];
        }
        Y_sum[] = sum;
    }
    
    // 3. Réduction globale pour obtenir min/max de la somme
    stats sY = statsf (Y_sum);
    
    // 4. Affichage des diagnostics
    fprintf(stderr, "DIAG i=%d | T:[%g, %g] | P:[%g, %g] | Y_sum:[%g, %g]\n", 
            i, sT.min, sT.max, sp.min, sp.max, sY.min, sY.max);
    fflush(stderr);

    // 5. Arrêt de sécurité strict
    // Si la température explose (>5000K) ou si la masse est non conservée (>1.1)
    /* if (isnan(sT.max) || sT.max > 5000.0 || sY.max > 1.1 || sY.min < 0.9) {
        fprintf(stderr, "FATAL: Divergence détectée à i=%d. Arrêt.\n", i);
        abort(); 
    }*/ 
}

event check_divergence (i++) {
    // Calcul de la divergence de la vitesse
    scalar divu[];
    foreach() {
        divu[] = (u.x[1,0] - u.x[-1,0] + u.y[0,1] - u.y[0,-1])/(2.*Delta);
    }
    stats sd = statsf(divu);
    fprintf(stderr, "DEBUG i=%d | div(u) : [%g, %g]\n", i, sd.min, sd.max);
}

void output_pvtu (scalar * slist, vector * vlist, char * prefix, double t) {
  // 1. Création des répertoires uniquement sur le Master (PID 0)
  if (pid() == 0) {
    // Note: on ignore les erreurs si le dossier existe déjà
    system("mkdir -p vtu/vtk_pieces"); 
  }
  
  #if _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  char vtu_name[128];
  sprintf (vtu_name, "vtu/vtk_pieces/%s_t_%.4f_n%d.vtu", prefix, t, pid());

  // Comptage des cellules locales (utilisant la réduction MPI automatique de Basilisk)
  long num_cells = 0;
  foreach(reduction(+:num_cells)) { 
    num_cells++; 
  }
  
  FILE * fp = fopen (vtu_name, "w");
  if (fp) {
    fprintf (fp, "<?xml version=\"1.0\"?>\n");
    fprintf (fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf (fp, "  <UnstructuredGrid>\n");
    fprintf (fp, "    <Piece NumberOfPoints=\"%ld\" NumberOfCells=\"%ld\">\n", num_cells * 4, num_cells);

    // --- Points (Coordonnées) ---
    fprintf (fp, "      <Points>\n");
    fprintf (fp, "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    foreach() {
      double hs = Delta / 2.0;
      fprintf (fp, "%g %g 0\n%g %g 0\n%g %g 0\n%g %g 0\n", 
               x - hs, y - hs, x + hs, y - hs, x + hs, y + hs, x - hs, y + hs);
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "      </Points>\n");
    
    // --- Cellules (Connectivité) ---
    fprintf (fp, "      <Cells>\n");
    fprintf (fp, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    for (long i = 0; i < num_cells * 4; i += 4) {
      fprintf (fp, "%ld %ld %ld %ld\n", i, i + 1, i + 2, i + 3);
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    for (long i = 1; i <= num_cells; i++) {
      fprintf (fp, "%ld\n", i * 4);
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for (long i = 0; i < num_cells; i++) {
      fprintf (fp, "9\n");
    }
    fprintf (fp, "        </DataArray>\n");
    fprintf (fp, "      </Cells>\n");
    
    // --- Données (Scalaires et Vecteurs) ---
    fprintf (fp, "      <CellData>\n");
    for (scalar s in slist) {
      fprintf (fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", s.name);
      foreach() fprintf (fp, "%g\n", s[]);
      fprintf (fp, "        </DataArray>\n");
    }
    for (vector v in vlist) {
      fprintf (fp, "        <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\" format=\"ascii\">\n", v.x.name);
      foreach() fprintf (fp, "%g %g 0\n", v.x[], v.y[]);
      fprintf (fp, "        </DataArray>\n");
    }
    fprintf (fp, "      </CellData>\n");
    fprintf (fp, "    </Piece>\n");
    fprintf (fp, "  </UnstructuredGrid>\n");
    fprintf (fp, "</VTKFile>\n");
    fclose (fp);
  }

  // --- Écriture du fichier maître (.pvtu) uniquement par le Master ---
  if (pid() == 0) {
    char pvtu_name[128];
    sprintf (pvtu_name, "vtu/%s_t_%.4f.pvtu", prefix, t);
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

      for (int i = 0; i < npe(); i++) {
        fprintf (fpvtu, "    <Piece Source=\"vtk_pieces/%s_t_%.4f_n%d.vtu\"/>\n", prefix, t, i);
      }
      fprintf (fpvtu, "  </PUnstructuredGrid>\n</VTKFile>\n");
      fclose (fpvtu);
    }
  }
}

event snapshot_vtu (t += 0.001; t <= T_END) {
  char prefix[80];
  sprintf (prefix, "fields"); 
  
  // Initialisation unique de la liste
  scalar * slist = NULL;
  
  // Ajout des scalaires T, P
  slist = list_append (slist, gas->T);
  slist = list_append (slist, p);
  
  // Ajout de toutes les espèces de Cantera
  for (scalar s in gas->YList) {
    slist = list_append (slist, s);
  }
  
  // Vecteurs
  vector * vlist = {u};
  output_pvtu (slist, vlist, prefix, t);
  
  // Libération de la mémoire de la liste
  free (slist);
}