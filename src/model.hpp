/*
 * This file is part of CELADRO-3D-CUDA, Copyright (C) 2024, Siavash Monfared
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <vector>
#include <array>
#include "vec_cuda.h"
#include "stencil.hpp"
#include "serialization.hpp"
#include "cuComplex.h"
#include <curand_kernel.h>

/** Type used to represent values on the grid */
using field = std::vector<double>;
/** Grid coordinate */
using coord = vec<unsigned, 3>;



/** Model class
 *
 * This class contains the whole program and is mainly used to be able to
 * scatter the implementation to different files without having to declare
 * every single variable extern. The architecture has not been really thought
 * through and one should not create two of these objects.
 * */
struct Model
{
  /** Simulation variables
   * @{ */

  /** List of neighbours
   *
   * The list of neighbours is used for computing the derivatives and is pre-
   * computed during initialization. The neighbours_patch variable has the same
   * role but for a region of the size of the cell patches (all patches have the
   * same size). These variables are computed in Initialize() and do not change
   * at runtime.
   * */
   
/** In which direction do we copy data? */
enum class CopyMemory {
	HostToDevice,
	DeviceToHost
};

/** Allocate or free memory? */
enum class ManageMemory {
	Allocate,
	Free
};

  std::vector<stencil> neighbors, neighbors_patch;
  /** Phase fields and derivatives */
  std::vector<field> phi, phi_dx, phi_dy, phi_dz;
  /** Predicted phi in a PC^n step */
  std::vector<field> phi_old;
  /** V = delta F / delta phi */
  std::vector<field> V;
  /** Sum of phi at each node */
  field sum_one, sum_two;
  /** Total polarization of the tissue */
  field field_polx, field_poly, field_polz;
  /** Total velocity of the tissue */
  field field_velx, field_vely, field_velz;
  /** Phase-field for the walls and their derivatives */
  field walls, walls_dx, walls_dy, walls_dz, walls_laplace;
  /** Forces */
  std::vector<vec<double, 3>> Fpol, Fnem, Fshape, Fpressure;
  /** cell stresses */
  std::vector<double> cSxx,cSxy,cSxz,cSyy,cSyz,cSzz;
  /** Velocity */
  std::vector<vec<double, 3>> velocity;
  /** vol associated with a phase field */
  std::vector<double> vol; 
  /** Polarisation */
  std::vector<vec<double, 3>> polarization;
  /** Polarisation total torque */
  std::vector<double> delta_theta_pol;
  /** Vorticity around each cell */
  std::vector<vec<double, 3>> vorticity;
  /** Stress tensor */
  field field_press;
  /** Phi difference */
  std::vector<field> dphi;
  /** Predicted phi difference in a P(C)^n step */
  std::vector<field> dphi_old;
  /** Direction of the polarisation */
  std::vector<double> theta_pol, theta_pol_old;
  /** Center-of-mass */
  std::vector<vec<double, 3>> com, com_prev;
  /** stress fields */
  field field_sxx, field_sxy, field_sxz, field_syy, field_syz, field_szz;
  /** heterogeneous cell properties */ 
  std::vector<double> stored_gam, stored_omega_cc, stored_omega_cs, stored_alpha, stored_dpol;
  /** Alignement options (see options.cpp) */
  int align_nematic_to = 0, align_polarization_to = 1;
  
  
  
  /** Sum_i S_i phi_i */
  //field sumS00, sumS01, sumS02, sumS12, sumS11, sumS22; 
  //field sumQ00, sumQ01;
  /** Structure tensor */
  //std::vector<double> S00, S01, S02, S12, S11, S22;
  /** Q-tensor */
  //std::vector<double> Q00, Q01;
  /** Direction of the nematics */
  //std::vector<double> theta_nem, theta_nem_old;
  /** Elastic torque for the nematic */
  //std::vector<double> tau;


  /** @} */

  /** Domain managment
   * @{ */

  /** Min of the boundaries of the patches and center of mass */
  std::vector<coord> patch_min;
  /** Max of the boundaries of the patches and center of mass */
  std::vector<coord> patch_max;
  /** Size of the patch (set by margin) */
  coord patch_size, patch_margin;
  /** Total number of nodes in a patch */
  unsigned patch_N;
  /** Memory offset for each patch */
  std::vector<coord> offset;
  /** Counter to compute com in Fourier space */
  std::vector<std::complex<double>> com_x, com_y, com_z;
  /** Precomputed tables for sin and cos (as complex numbers) used in the
   * computation of the com.
   * */
  std::vector<std::complex<double>> com_x_table, com_y_table, com_z_table;

  /** @} */

  /** Program options
   * @{ */

  /** verbosity level
   *
   * 0: no output
   * 1: some output
   * 2: extended output (default)
   * 3: debug
   * */
  unsigned verbose = 2;
  /** compress output? (we use zip) */
  bool compress, compress_full;
  /** name of the run */
  std::string runname;
  /** Output dir (or tmp dir before moving files to the archive) */
  std::string output_dir;
  /** write any output? */
  bool no_write = false;
  /** skip runtime warnings? */
  bool no_warning = false;
  /** are the runtime warnings fatal? (i.e. they do stop the simulation) */
  bool stop_at_warning = false;
  /** shall we perform runtime checks ? */
  bool runtime_check = false;
  /** shall we print some runtime stats ? */
  bool runtime_stats = false;
  /** padding for onscreen output */
  unsigned pad;
  /** name of the inpute file */
  std::string inputname = "";
  /** Delete output? */
  bool force_delete;
  /** The random number seed */
  unsigned long seed; // picked using a fair die
  /** Flag if seed was set in the arguments, see options.hpp */
  bool set_seed;
  /** Number of predictor-corrector steps */
  unsigned npc = 1;
  /** Relaxation time at initialization */
  unsigned relax_time = 0;
  /** Value of nsubstep to use for initialization */
  unsigned relax_nsubsteps = 0;
  /** Total time spent writing output */
  std::chrono::duration<double> write_duration;
  /** @} */

  /** Simulation parameters
   * @{ */

  /** size of the system */
  vec<unsigned, 3> Size;
  /** total number of nodes */
  unsigned N;
  /** Total number of time steps */
  unsigned nsteps;
  /** Time interval between data outputs */
  unsigned ninfo = 10;
  /** Time at which to start the output */
  unsigned nstart = 0;
  /** number of subdivisions for a time step */
  unsigned nsubsteps = 10;
  /** effective time step */
  double time_step, sqrt_time_step;
  /** Number of phases */
  unsigned nphases;
  /** angle in degrees (input variable only) */
  double angle_deg;
  /** Margin for the definition of patches */
  unsigned margin = 25;
  /** Boudaries for cell generation
   *
   * These are the boundaries (min and max x and y components) of the domain in
   * which the cells are created when the initial config 'random' is choosen.
   * */
  std::vector<unsigned> birth_bdries;
  /** @} */

  /** Cell properties
   * @{ */

  /** Elasticity */
  double gam = 0.008;
  /** Energy penalty for vol */
  double mu = 45;
  /** Interface thickness */
  double lambda = 3;
  /**  Interaction stength */
  double kappa_cc = 0.5;
  /** cell-cell Adhesion */
  double omega_cc = 0.0005;
  /** cell-substrate adhesion */
  double omega_cs = 0.0008;
  /** Activity from shape */
  double zetaS = 0, sign_zetaS = 0;
  /** Activity from internal Q tensor */
  double zetaQ = 0, sign_zetaQ = 0;
  /** Propulsion strength */
  double alpha = 0.05;
  /** Substrate friction parameter */
  double xi = 1;
  /** Prefered radii (vol = pi*R*R) and radius growth */
  double R = 8;
  /** Repuslion by the wall */
  double kappa_cs = 0.5;
  /** Adhesion on the wall */
  // double wall_omega = 0;
  /** Elasitc parameters */
  double Knem = 0, Kpol = 0;
  /** Strength of polarity / nematic tensor */
  double Spol = 0, Snem = 0;
  /** Flow alignment strenght */
  double Jpol = 0, Jnem = 0;
  /** Vorticity coupling */
  double Wnem = 0;
  /** Noise strength */
  double Dpol = 0, Dnem = 0;
  double vimp = (4./3.)*Pi*R*R*R;
  unsigned globalT = 0;
  
  // ===========================================================================
  // related to proliferation 
  struct cellInfo{
  double birth_time;
  double death_time;
  int parent;
  double physicalprop;
  int generation;
  };
  
  void stress_criterionOU(unsigned, bool&, double&);
  double UpdateOU(double tcurrent, double tmean, double tcorr, double sigma, const double dt);
  void initDivisionOU(unsigned n, unsigned i, double angle, unsigned t, bool mutate);
  std::vector<double> timer, divisiontthresh, stored_tmean;
  double tcorr, tmean, sigma;
  double max_prop_val;
  double min_prop_val;
  std::map<int, cellInfo> cellHist;
  void cellLineage(int cell_id, int parent_id, double birth_t, double death_t, double physicalprop, int gen);
  std::vector<unsigned> nphases_index;
  int tau_divide = 0;
  double mutation_strength = 0.;
  bool proliferate_bool = true;
  unsigned prolif_start;
  double prolif_freq_mean, prolif_freq_std;
  double scaling_factor = 5.;
  unsigned nphases_init;
  unsigned nphases_max = 1000;
  unsigned nphases_index_head;
  unsigned GlobalCellIndex;
  void proliferate(unsigned);
  void initDivision(unsigned n, unsigned i, double angle, unsigned t);
  void BirthCellMemories(unsigned new_nphases);
  void DivideCell(unsigned n, unsigned nphases_current, double angle, double cellProp);
  void BirthCell(unsigned n);
  void ComputeBirthCellCOM(unsigned n, unsigned nbirth);
  void KillCell(unsigned n, unsigned i);
  void BirthCellAtNode(unsigned n, unsigned q);
  void print_new_cell_props();
  void AllocDeviceMemoryCellBirth();
  void FreeDeviceMemoryCellBirth();
  void _manage_device_memoryCellBirth(ManageMemory);
  void Write_divAngle(unsigned t,unsigned n,unsigned i, bool mutate,double angle);
  std::vector<double> compute_eigen(double sxx,double sxy, double syy);
  std::vector<double> stress_criterion();
  void write_cellHist_binary(const std::string &filename,
                               unsigned currentTime,
                               const std::map<int, cellInfo> &hist);
				  
  // ===========================================================================
  // Options. Implemented in options.cpp

  /** the variables map used to collect options */
  opt::variables_map vm;

  /** Set parameters from input */
  void ParseProgramOptions(int ac, char **av);

  /** Output all program options */
  void PrintProgramOptions();

  // =========================================================================
  // Program managment. Implemented in main.cpp

  /** The main loop */
  void Algorithm();

  /** Setup computation */
  void Setup(int, char**);

  /** Do the computation */
  void Run();

  /** Clean after you */
  void Cleanup();

  // ===========================================================================
  // Configuration. Implemented in configure.cpp

  /** Initial configuration parameters
   * @{ */

  /** Initial configuration name */
  std::string init_config;
  /** Boundary conditions flag */
  unsigned BC = 0;
  /** Noise level for the nematic tensor initial configuration */
  double noise = 1;
  /** Wall thickness */
  double wall_thickness = 1.;
  /** Ratio of the cross vs size of the domain (BC=4) */
  double cross_ratio = .25;
  /** Ratio of the wound vs size of the domain (BC=5) */
  double wound_ratio = .50;
  /** Ratio of the tumor vs size of the domain (BC=6) */
  double tumor_ratio = .80;

  /** @} */

  /** Add cell with number n at a certain position */
  void AddCell(unsigned n, const coord& center);
  void AddCellMix(unsigned n, const coord& center);

  /** Subfunction for AddCell() */
  void AddCellAtNode(unsigned n, unsigned q, const coord& center);

  /** Set initial condition for the fields */
  void Configure();

  /** Set initial configuration for the walls */
  void ConfigureWalls(int BC);

  // ==========================================================================
  // Writing to file. Implemented in write.cpp

  /** Write current state of the system */
  void WriteFrame(unsigned);
  
  /** Write phase-field for cell n */
  void Write_phi(unsigned);
  void Write_dphi(unsigned);
  void Write_COM(unsigned);
  void Write_velocities(unsigned);    
  void Write_forces(unsigned);    
  void Write_visData(unsigned);
  void Write_contArea(unsigned);
  void Write_Density(unsigned);
  void visTMP(unsigned);
  void Write_OU(unsigned,unsigned);
  
  /** Write run parameters */
  void WriteParams();

  /** Remove old files */
  void ClearOutput();

  /** Create output directory */
  void CreateOutputDir();

  // ==========================================================================
  // Initialization. Implemented in init.cpp

  /** Initialize memory for field */
  void Initialize();

  /** Allocate memory for individual cells */
  void SetCellNumber(unsigned new_nphases);

  /** Initialize neighbors list (stencils) */
  void InitializeNeighbors();

  /** Swap two cells in the internal arrays */
  void SwapCells(unsigned n, unsigned m);

  // ===========================================================================
  // Random numbers generation. Implemented in random.cpp

  /** Pseudo random generator */
  std::mt19937 gen;
  //ranlux24 gen;
  
  /** Return random real, uniform distribution */
  double random_real(double min=0., double max=1.);
  /** Return random real, uniform distribution */
  double random_uniform();
  /** Return random real, gaussian distributed */
  double random_normal(double sigma=1.);
  /** Return random real, gaussian distributed */
  double random_normal_full(double mean=0., double sigma=1.);
  /** Return random uniform double */
  double random_double_uniform(double min, double max);
  /** Return random longnormal */ 
  double random_lognormal(double mu, double sig, double minval);
  double random_lognormal(double mu, double sig);

  /** Return geometric dist numbers, prob is p */
  unsigned random_geometric(double p);

  /** Return poisson distributed unsigned integers */
  unsigned random_poisson(double lambda);

  /** Return exp distributed unsigned integers */
  int random_exponential(double lambda);

  /** Return random unsigned uniformly distributed */
  unsigned random_unsigned();
  int random_int_uniform(int min, int max);


  /** Initialize random numbers
   *
   * If CUDA is enabled need to intialize CUDA random numbers
   * */
  void InitializeRandomNumbers();

  // ==========================================================================
  // Support for cuda. Implemented in cuda.cu

  /** Device(s) propeties
   * @{ */

  /** Obtain (and print) device(s) properties
   *
   * Also checks that the device properties are compatible with the values given
   * in src/cuda.h.
   * */
  void QueryDeviceProperties();

  /** @} */

  /** Pointer to device global memory
   *
   * These pointers reflects the program data strcuture and represents the cor-
   * responding data on the device global memory. All names should be identical
   * to their host counterparts apart from the d_ prefix.
   *
   * @{ */
    
  double *d_phi, *d_phi_old, *d_V, *d_phi_dx, *d_phi_dy, *d_phi_dz, *d_dphi, *d_dphi_old,
         *d_walls, *d_walls_laplace, *d_walls_dx, *d_walls_dy, *d_walls_dz, *d_vol,
         *d_theta, *d_sum_one, *d_sum_two,*d_field_velx, *d_field_vely, *d_field_velz, 
         *d_field_polx, *d_field_poly, *d_field_polz, *d_field_press, *d_delta_theta_pol,
         *d_theta_pol, *d_theta_pol_old, *d_field_sxx, *d_field_sxy, *d_field_sxz, *d_field_syy,
         *d_field_syz, *d_field_szz, *d_cSxx, *d_cSxy, *d_cSxz, *d_cSyy, *d_cSyz, *d_cSzz,
         *d_stored_gam, *d_stored_omega_cc, *d_stored_omega_cs, *d_stored_alpha, *d_stored_dpol;
  vec<double, 3>  *d_polarization, *d_velocity, *d_Fpol, *d_Fpressure, *d_vorticity, *d_com;
  stencil         *d_neighbors, *d_neighbors_patch;
  coord           *d_patch_min, *d_patch_max, *d_offset;
  cuDoubleComplex *d_com_x, *d_com_y, *d_com_z, *d_com_x_table, *d_com_y_table, *d_com_z_table;
  
  /** @} */

  /** Random number generation
   * @{ */

  /** Random states on the device */
  curandState *d_rand_states;

  /** Initialization function */
  void InitializeCuda();

  /** Initialization function for random numbers */
  void InitializeCUDARandomNumbers();

  /** @} */
  /** CUDA device memory managment
    * @{ */



  /** Implementation for AllocDeviceMemory() and FreeDeviceMemory() */
  void _manage_device_memory(ManageMemory);
  /** Implementation for PutToDevice() and GetFromDevice() */
  void _copy_device_memory(CopyMemory);

  /** Copy data to the device global memory
   *
   * This function is called at the begining of the program just before the main
   * loop but after the system has been initialized.
   * */
  void PutToDevice();

  /** Copy data from the device global memory
   *
   * This function is called every time the results need to be dumped on the disk.
   * */
  void GetFromDevice();

  /** Allocate memory for all device arrays */
  void AllocDeviceMemory();

  /** Allocate memory for all device arrays */
  void FreeDeviceMemory();

  /** @} */

  /** Runtime properties
   * @{ */

  int n_total, n_blocks, n_threads;
  int nph_total, nph_blocks, nph_threads;

  /** @} */

  // ===========================================================================
  // Run. Implemented in run.cpp

  /** Time step
   *
   * This is the time-stepping function and performs the main computation.
   * */
  void Step();

  /** Prepare before run */
  void Pre();

  /** Prints some stats before running */
  void PreRunStats();

  /** Prints some stats in between ninfo steps */
  void RuntimeStats();

  /** Performs punctual check at runtime */
  void RuntimeChecks();

  /** Post run function */
  void Post();

  double SubAdhesion_field(double,double);
  /** Subfunction for update */    
  void UpdatePotAtNode(unsigned, unsigned);

  /** Subfunction for update */
  void UpdatePhaseFieldAtNode(unsigned, unsigned, bool);

  /** Subfunction for update */
  void UpdateForcesAtNode(unsigned, unsigned);

  /** Subfunction for update */
  void UpdateStructureTensorAtNode(unsigned, unsigned);

  /** Subfunction for update */
  void UpdateSumsAtNode(unsigned, unsigned);

  /** Subfunction for update */
  void ReinitSumsAtNode(unsigned);

  /** Compute center of mass of a given phase field */
  void ComputeCoM(unsigned);

  /** Update polarisation of a given field */
  void UpdatePolarization(unsigned, bool);

  /** Update nematic tensor of a given field */
  void UpdateNematic(unsigned, bool);

  /** Compute shape parameters
   *
   * This function effectively computes the second moment of vol, which ca n be used to
   * fit the shape of a cell to an ellipse.
   * */
  void ComputeShape(unsigned);

  /** Update the moving patch following each cell */
  void UpdatePatch(unsigned);

  /** Update fields
   *
   * The boolean argument is used to differentiate between the predictor step
   * (true) and subsequent corrector steps.
   * */
  void Update(bool, unsigned t);

  // ===========================================================================
  // Serialization

  /** Serialization of parameters (in and out) */
  template<class Archive>
  void SerializeParameters(Archive& ar)
  {
    ar & auto_name(gam)
       & auto_name(mu)
       & auto_name(lambda)
       & auto_name(nphases)
       & auto_name(kappa_cc)
       & auto_name(xi)
       & auto_name(R)
       & auto_name(alpha)
       & auto_name(zetaS)
       & auto_name(zetaQ)
       & auto_name(omega_cc)
       & auto_name(wall_thickness)
       & auto_name(kappa_cs)
       & auto_name(omega_cs)
       & auto_name(walls)
       & auto_name(patch_margin)
       & auto_name(relax_time)
       & auto_name(relax_nsubsteps)
       & auto_name(npc)
       & auto_name(seed)
       & auto_name(Knem)
       & auto_name(Kpol)
       & auto_name(Snem)
       & auto_name(Spol)
       & auto_name(Jnem)
       & auto_name(Jpol)
       & auto_name(Wnem)
       & auto_name(Dpol)
       & auto_name(Dnem)
       & auto_name(margin)
       & auto_name(patch_size);
  }

  /** Serialization of parameters (in and out) */
  template<class Archive>
  void SerializeFrame(Archive& ar)
  {
    ar & auto_name(nphases)
       & auto_name(phi)
       & auto_name(field_sxx)
       & auto_name(field_syy)
       & auto_name(field_szz)
       & auto_name(field_sxy)
       & auto_name(field_sxz)
       & auto_name(field_syz)
       & auto_name(stored_gam)
       & auto_name(stored_omega_cc)
       & auto_name(stored_omega_cs)
       & auto_name(stored_alpha)
       & auto_name(stored_dpol)
       & auto_name(cSxx)
       & auto_name(cSxy)
       & auto_name(cSxz)
       & auto_name(cSyy)
       & auto_name(cSyz)
       & auto_name(cSzz)
       & auto_name(offset)
       & auto_name(com)
       & auto_name(velocity)
       & auto_name(Fpol)
       & auto_name(Fpressure)
       & auto_name(theta_pol)
       & auto_name(patch_min)
       & auto_name(patch_max);
  }
  
  // ===========================================================================
  // Tools
  
  /** Gives domain coordinate corresponding to a domain index */
  coord GetPosition(unsigned k) const
  { return { GetXPosition(k), GetYPosition(k), GetZPosition(k) }; }

  /** Gives domain x-coordinate corresponding to a domain index */
  unsigned GetXPosition(unsigned k) const
  { return (k/Size[1])%Size[0];}

  /** Gives domain y-coordinate corresponding to a domain index */
  unsigned GetYPosition(unsigned k) const
  {return k%Size[1];}
  
  unsigned GetZPosition(unsigned k) const 
  {return k/(Size[0]*Size[1]);}
    
  /** Get domain index from domain coordinates */
  unsigned GetIndex(const coord& p) const
  {return p[1] + Size[1]*p[0] + Size[0]*Size[1]*p[2];}

  /** Get patch index from domain coordinates */
  unsigned GetPatchIndex(unsigned n, coord p) const
  {
    // get difference to the patch min
    p = (p + Size - patch_min[n])%Size;
    // correct for offset
    p = (p + patch_size - offset[n])%patch_size;
    // remap linearly
    return p[1] + patch_size[1]*p[0] + patch_size[0]*patch_size[1]*p[2];
  }
  
  /** Get patch index from domain index */
  unsigned GetPatchIndex(unsigned n, unsigned k) const
  {
    return GetPatchIndex(n, GetPosition(k));
  }

  /** Get domain index from patch index */
  unsigned GetIndexFromPatch(unsigned n, unsigned q) const
  {
    // position on the patch
    const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    // position on the domain
    const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    // return domain index
    return GetIndex(dpos);
  }
  
  coord GetNodePosOnPatch(unsigned n, unsigned q) const
  {
    // position on the patch
    const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    return qpos;
  }  
  
  coord GetNodePosOnDomain(unsigned n, unsigned q) const
  {
    // position on the patch
    const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    // position on the domain
    const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    return dpos;
  }    
  
};

#endif//MODEL_HPP_
