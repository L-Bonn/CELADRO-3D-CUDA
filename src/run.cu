/*
 * This file is part of CELADRO-3D-CUDA, Copyright (C) 2024, Siavash Monfared
 *
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

 
#include "header.hpp"
#include "model.hpp"
#include "derivatives.hpp"
#include "tools.hpp"
#include "cuda.h"
//#include "reduce.h"
#include "cuComplex.h"
#include "curand_kernel.h"
#include <math.h>  // For atan2

using namespace std;

// Define a device (and host) inline function to compute the argument
__host__ __device__ inline double cuCarg(cuDoubleComplex z)
{
    return atan2(cuCimag(z), cuCreal(z));
}


void Model::Pre()
{

  if(relax_time>0)
  {

    double save_alpha  = 0; swap(alpha,  save_alpha);
    double save_zetaS  = 0; swap(zetaS,  save_zetaS);
    double save_zetaQ  = 0; swap(zetaQ,  save_zetaQ);
    double save_Dnem  = 0; swap(Dnem,  save_Dnem);
    double save_Dpol  = 0; swap(Dpol,  save_Dpol);
    double save_Jnem  = 0; swap(Jnem,  save_Jnem);
    double save_Jpol  = 0; swap(Jpol,  save_Jpol);
    double save_Kpol  = 0; swap(Kpol,  save_Kpol);
    double save_Knem  = 0; swap(Knem,  save_Knem);
    double save_Wnem  = 0; swap(Wnem,  save_Wnem);

    if(relax_nsubsteps) swap(nsubsteps, relax_nsubsteps);

    for(unsigned i=0; i<relax_time*nsubsteps; ++i)
      for(unsigned j=0; j<=npc; ++i) Update(i==0);

    if(relax_nsubsteps) swap(nsubsteps, relax_nsubsteps);

    swap(alpha, save_alpha);
    swap(zetaS, save_zetaS);
    swap(zetaQ, save_zetaQ);
    swap(Jnem, save_Jnem);
    swap(Jpol, save_Jpol);
    swap(Dnem, save_Dnem);
    swap(Dpol, save_Dpol);
    swap(Kpol, save_Kpol);
    swap(Knem, save_Knem);
    swap(Wnem, save_Wnem);

  }

  if(BC==5 || BC==7) ConfigureWalls(1);
  if(BC==6) ConfigureWalls(0);
}

void Model::Post()
{}

void Model::PreRunStats()
{}

void Model::RuntimeStats()
{}

void Model::RuntimeChecks()
{}

__global__
void cuUpdateSumsAtNode(	   double *phi,
				   double *sum_one, 
				   double *sum_two,
				   double *field_polx,
				   double *field_poly,
				   double *field_polz,
				   double *field_velx,
				   double *field_vely,
				   double *field_velz,
				   vec<double,3> *polarization, 
				   vec<double,3> *velocity,
				   unsigned patch_N,
				   int n_total,
				   coord patch_size,
				   coord *patch_min,
				   coord Size,
				   coord *offset)

				   
{

	// build indices with cuda!!
	const int m = blockIdx.x*blockDim.x + threadIdx.x;
	if(m>=n_total) return;
	
	const unsigned n = static_cast<unsigned>(m)/patch_N;
	unsigned q = static_cast<unsigned>(m)%patch_N;

	const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    	
    	const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    	
    	const auto k = dpos[1] + Size[1]*dpos[0] + Size[0]*Size[1]*dpos[2];
	double p = phi[m];

	/*
	atomicAdd(&sum_one[k],p);
	atomicAdd(&sum_two[k],p*p);
	atomicAdd(&field_polx[k],p*polarization[n][0]);
	atomicAdd(&field_poly[k],p*polarization[n][1]);
	atomicAdd(&field_polz[k],p*polarization[n][2]);
	atomicAdd(&field_velx[k],p*velocity[n][0]);
	atomicAdd(&field_vely[k],p*velocity[n][1]);
	atomicAdd(&field_velz[k],p*velocity[n][2]);
	*/
	sum_one[k] += p;
	sum_two[k] += p*p;
	field_polx[k] += p*polarization[n][0];
	field_poly[k] += p*polarization[n][1];
	field_polz[k] += p*polarization[n][2];
	field_velx[k] += p*velocity[n][0];
	field_vely[k] += p*velocity[n][1];
	field_velz[k] += p*velocity[n][2];
	
	
}


__global__	
void cuUpdatePotAtNode(stencil *neighbors,
				  stencil *neighbors_patch,
				  double *phi,
				  double *sum_one, 
				  double *sum_two, 
				  double *walls,
				  double *walls_laplace,
				  double *field_press,
				  double *V,
				  double *vol,
				  coord patch_size,
				  coord *patch_min,
				  coord Size,
				  coord *offset,
				  vec<double,3> *Fpol,
				  vec<double,3> *Fpressure,
				  vec<double,3> *vorticity,
				  double *delta_theta_pol,
				  double kappa_cc,
				  double mu,
				  double lambda,
				  double gam,
				  double vimp,
				  double omega_cc,
				  double omega_cs,
				  double kappa_cs,
				  int n_total,
				  unsigned patch_N,
				  double *field_sxx,
				  double *field_sxy,
				  double *field_sxz,
				  double *field_syy,
				  double *field_syz,
				  double *field_szz,
				  double xi,
				  double *field_velx,
				  double *field_vely,
				  double *field_velz)
{

	// build indices with cuda!!
	const int m = blockIdx.x*blockDim.x + threadIdx.x;
	if(m>=n_total) return;
	
	const unsigned n = static_cast<unsigned>(m)/patch_N;
	unsigned q = static_cast<unsigned>(m)%patch_N;
	const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    	const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    	const auto k = dpos[1] + Size[1]*dpos[0] + Size[0]*Size[1]*dpos[2];
	double p = phi[m];

	// update potential (internal + interactions) 
	const auto& s  = neighbors[k];
	const auto& sq = neighbors_patch[q];	
	const auto ll = laplacian(&phi[n*patch_N], sq);
	const auto ls = laplacian(sum_one, s);

	const double internal = (
	+ gam*(8*p*(1-p)*(1-2*p)/lambda - 2*lambda*ll)
	- 4*mu/vimp*(1-vol[n]/vimp)*p
	);

	const double interactions = (
	// repulsion term
	+ 2*kappa_cc/lambda*p*(sum_two[k]-p*p)
	// adhesion term
	- 2*omega_cc*lambda*(ls-ll)
	// repulsion with walls
	+ 2*kappa_cs/lambda*p*walls[k]*walls[k]
	// adhesion with walls
	- 2*omega_cs*lambda*walls_laplace[k]
	);
	
	// delta F / delta phi_i
	V[m] = internal + interactions;
	// pressure
	// atomicAdd(&field_press[k],p*interactions);
	field_press[k] += p*interactions;
	
	// -----------------------------------------------------------------------------
	// compute stress field 
	// -----------------------------------------------------------------------------
	
	// const int nsite = 2;
	double factor = 8;
	int nvx = Size[0]-1;
	int nvy = Size[1]-1;
	int nvz = Size[2]-1;
       double cx = (k/Size[1])%Size[0] + 0.5;
       double cy = k%Size[1] + 0.5;
       double cz = k/(Size[0]*Size[1]) + 0.5;
       // Loop over the 2×2×2 integration points (unrolled for clarity)
       for (int dz = 0; dz < 2; dz++) {
       	for (int dy = 0; dy < 2; dy++) {
              	for (int dx = 0; dx < 2; dx++) {
              		double ix = (cx-0.5)+dx;
              		double iy = (cy-0.5)+dy;
              		double iz = (cz-0.5)+dz;
              		double diff_x = cx - ix;
              		double diff_y = cy - iy;
              		double diff_z = cz - iz;
              		double norm = sqrt(diff_x*diff_x+diff_y*diff_y+diff_z*diff_z);
              		if (norm == 0) norm = 1;
              		double ux = diff_x / norm;
              		double uy = diff_y / norm;
              		double uz = diff_z / norm;
              		int idx = iy + Size[1]*ix + Size[0]*Size[1]*iz;
              		field_sxx[k] += ux*xi*field_velx[idx];
              		field_sxy[k] += ux*xi*field_vely[idx];
              		field_sxy[k] += uy*xi*field_velx[idx];
              		field_sxz[k] += ux*xi*field_velz[idx];
              		field_sxz[k] += uz*xi*field_velx[idx];
              		field_syy[k] += uy*xi*field_vely[idx];
              		field_syz[k] += uy*xi*field_velz[idx];
              		field_syz[k] += uz*xi*field_vely[idx];
              		field_szz[k] += uz*xi*field_velz[idx];
              	}
              }
       }
       field_sxx[k] /= factor;
       field_sxy[k] /= (2.*factor);
       field_sxz[k] /= (2.*factor);
       field_syy[k] /= factor;
       field_syz[k] /= (2.*factor);
       field_szz[k] /= factor;
       

	if (q==0){
    	Fpol[n] = Fpressure[n] = vorticity[n] = {0, 0, 0};//add fnem[n],fshape[n] --> 		  cell_fpressure[n],cell_pol[n],cell_delta_theta_pol
    	delta_theta_pol[n] = 0;// add tau[n]
	}
	
}

__global__
void cuUpdatePhysicalFieldsAtNode( stencil *neighbors,
					  	stencil *neighbors_patch,
					  	double *phi, 
					  	double *phi_dx,
					  	double *phi_dy,
					  	double *phi_dz,
				  		double *field_press,
					  	double *sum_one, 
					  	double *vol,
					  	cuDoubleComplex *com_x,
					  	cuDoubleComplex *com_y,
					  	cuDoubleComplex *com_z,
					  	double *P0,
					  	double *P1,
					  	double *P2,
					  	double *U0,
					  	double *U1,
					  	double *U2,
				  		vec<double,3> *Fpressure,
				  		vec<double,3> *vorticity,
				  		double *delta_theta_pol,
				  		vec<double,3> *polarization,
					  	coord patch_size,
					  	coord *patch_min,
					  	coord Size,
					  	coord *offset,
					  	int n_total,
					  	unsigned patch_N,
						double *field_sxx,
						double *field_sxy,
						double *field_sxz,
						double *field_syy,
						double *field_syz,
						double *field_szz,
						double *cSxx,
						double *cSxy,
						double *cSxz,
						double *cSyy,
						double *cSyz,
						double *cSzz)		  	
{

	// build indices with cuda!!
	const int m = blockIdx.x*blockDim.x + threadIdx.x;
	if(m>=n_total) return;
	const unsigned n = static_cast<unsigned>(m)/patch_N;
	unsigned q = static_cast<unsigned>(m)%patch_N;
	const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    	const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    	const auto k = dpos[1] + Size[1]*dpos[0] + Size[0]*Size[1]*dpos[2];
	// double p = phi[m];

	const auto& s  = neighbors[k];
	const auto& sq = neighbors_patch[q];

	const auto dx  = derivX(&phi[n*patch_N], sq);
	const auto dy  = derivY(&phi[n*patch_N], sq);
	const auto dz  = derivZ(&phi[n*patch_N], sq);

	const auto dxs = derivX(sum_one, s);
	const auto dys = derivY(sum_one, s);
	const auto dzs = derivZ(sum_one, s);
	  
	const vec<double,3> fpressure = { field_press[k]*dx, field_press[k]*dy, field_press[k]*dz };//-->field_press[k]

	/*
	double fshape    = { zetaS_field[n]*sumS00[k]*dx + zetaS_field[n]*sumS01[k]*dy + zetaS_field[n]*sumS02[k]*dz,
		      zetaS_field[n]*sumS01[k]*dx + zetaS_field[n]*sumS11[k]*dy + zetaS_field[n]*sumS12[k]*dz,
		      zetaS_field[n]*sumS02[k]*dx + zetaS_field[n]*sumS12[k]*dy + zetaS_field[n]*sumS22[k]*dz };
		      
	double fnem      = { zetaQ_field[n]*sumQ00[k]*dx + zetaQ_field[n]*sumQ01[k]*dz, 
		     zetaQ_field[n]*sumQ00[k]*dy + zetaQ_field[n]*sumQ01[k]*dz,
		     zetaQ_field[n]*sumQ01[k]*dx - zetaQ_field[n]*sumQ00[k]*dz }; 
	*/
		  
	atomicAdd(&Fpressure[n][0],fpressure[0]);//-->Fpressure[n]
	atomicAdd(&Fpressure[n][1],fpressure[1]);//-->Fpressure[n]
	atomicAdd(&Fpressure[n][2],fpressure[2]);//-->Fpressure[n]
	//atomicAdd(&cell_fshape,fshape);
	//atomicAdd(&cell_fnem,fnem);
	
	atomicAdd(&cSxx[n],field_sxx[k]);
	atomicAdd(&cSxy[n],field_sxy[k]);
	atomicAdd(&cSxz[n],field_sxz[k]);
	atomicAdd(&cSyy[n],field_syy[k]);
	atomicAdd(&cSyz[n],field_syz[k]);
	atomicAdd(&cSzz[n],field_szz[k]);
	
	// store derivatives
	phi_dx[m] = dx;
	phi_dy[m] = dy;
	phi_dz[m] = dz;

	// nematic torques
	// tau[n]       += phi[n][q] * (sumQ00[k]*Q01[n] - sumQ01[k]*Q00[n]);
	// vorticity
	const vec<double,3> vortval = { U2[k]*dy-U1[k]*dz, U0[k]*dz-U2[k]*dx, U2[k]*dx-U0[k]*dy };//--> field_velx
	atomicAdd(&vorticity[n][0],-vortval[0]);//--> vorticity[n]
	atomicAdd(&vorticity[n][1],-vortval[1]);//--> vorticity[n]
	atomicAdd(&vorticity[n][2],-vortval[2]);//--> vorticity[n]
	// polarization torques
	const double ovlap = -( dx*(dxs-dx) + dy*(dys-dy) + dz*(dzs-dz)  );
	const vec<double, 3> P = { P0[k]-phi[m]*polarization[n][0], P1[k]-phi[m]*polarization[n][1], P2[k]-phi[m]*polarization[n][2] };//-->field_polx ... 

	const double delt_theta_pol= ovlap*atan2( 
	sqrt(pow( (P[1]*polarization[n][0]-P[0]*polarization[n][1]) ,2) + pow( (P[2]*polarization[n][0]-P[0]*polarization[n][2]) ,2) + pow( (P[2]*polarization[n][1]-P[1]*polarization[n][2]) ,2) ),
	P[0]*polarization[n][0]+P[1]*polarization[n][1]+P[2]*polarization[n][2]                               
		                     );
	atomicAdd(&delta_theta_pol[n],delt_theta_pol);//--> delta_theta_pol[n]
	
	if(q==0){
	com_x[n] = com_y[n] = com_z[n] = make_cuDoubleComplex(0., 0.);
	vol[n] = 0.;
	//S00[n] = S01[n] = S02[n] = S12[n] = S11[n] = S22[n] = vol[n] = 0;
	}

}

/*
__global__
void cuUpdatePolVel(
    double alpha,
    double xi,
    vec<double,3> *Fpressure,
    vec<double,3> *Fpol,
    vec<double,3> *velocity,
    vec<double,3> *polarization,
    // Removed: double *delta_theta_pol,
    coord patch_size,
    coord *patch_min,
    coord Size,
    coord *offset,
    int n_total,
    unsigned patch_N)
{
    // Build indices with CUDA.
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if(m >= n_total) return;
    unsigned n = static_cast<unsigned>(m) / patch_N;
    unsigned q = static_cast<unsigned>(m) % patch_N;
    
    const coord qpos = { (q / patch_size[1]) % patch_size[0],
                         q % patch_size[1],
                         q / (patch_size[0] * patch_size[1]) };
    const coord dpos = ( (qpos + offset[n]) % patch_size + patch_min[n] ) % Size;
    const auto k = dpos[1] + Size[1] * dpos[0] + Size[0] * Size[1] * dpos[2];

    if(q == 0) {
        // Update structure for the cell.
        Fpol[n]     = alpha * polarization[n];
        velocity[n] = (Fpressure[n] + Fpol[n]) / xi;  // add Fnem, Fshape if needed
    }
}
*/

__global__
void cuUpdatePolVel(
    double alpha,
    double xi,
    vec<double,3> *Fpressure,
    vec<double,3> *Fpol,
    vec<double,3> *velocity,
    vec<double,3> *polarization,
    int n_total)
{
	const int m = blockIdx.x * blockDim.x + threadIdx.x;
	if(m >= n_total) return;
	Fpol[m]     = alpha * polarization[m];
	velocity[m] = (Fpressure[m] + Fpol[m]) / xi; //add nematic+shape...
}

__global__
void cuUpdatePhaseFieldAtNode(	double *phi, 
					  	double *phi_dx,
					  	double *phi_dy,
					  	double *phi_dz,
					  	double *dphi,
					  	double *dphi_old,
					  	double *phi_old,
					  	double *sum_one, 
					  	double *sum_two,
					  	double *field_press,
					  	double *field_velx,
					  	double *field_vely,
					  	double *field_velz,
					  	double *vol,
					  	double *V,
					  	cuDoubleComplex *com_x,
					  	cuDoubleComplex *com_y,
					  	cuDoubleComplex *com_z,
					  	double *theta_pol,
					  	double *theta_pol_old,
					  	double time_step,
					  	cuDoubleComplex *com_x_table,
					  	cuDoubleComplex *com_y_table,
					  	cuDoubleComplex *com_z_table,
					  	double alpha,
					  	double xi,
				  		vec<double,3> *Fpressure,
				  		vec<double,3> *Fpol,
				  		vec<double,3> *velocity,
				  		vec<double,3> *polarization,
				  		vec<double,3> *com,
				  		double *delta_theta_pol,
					  	coord patch_size,
					  	coord *patch_min,
					  	coord *patch_max,
					  	coord patch_margin,
					  	coord Size,
					  	coord *offset,
					  	int n_total,
					  	unsigned patch_N,
					  	double Spol,
					  	double Dpol,
					  	double Kpol,
					  	double Jpol,
					  	unsigned N,
					  	curandState *rand_states,
					  	bool store)
					  	/*
					  	double *field_polx,
						double *field_poly,
						double *field_polz)
						*/
{

	
	// build indices with cuda!!
	const int m = blockIdx.x*blockDim.x + threadIdx.x;
	if(m>=n_total) return;
	unsigned n = static_cast<unsigned>(m)/patch_N;
	unsigned q = static_cast<unsigned>(m)%patch_N;
	
	const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    	const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    	const auto k = dpos[1] + Size[1]*dpos[0] + Size[0]*Size[1]*dpos[2];
	double p = phi[m];

  dphi[m] =
    - V[m]
    - velocity[n][0]*phi_dx[m] - velocity[n][1]*phi_dy[m] - velocity[n][2]*phi_dz[m];
    ;


  if(store)
  {
    dphi_old[m] = dphi[m];
    phi_old[m]  = phi[m];
  }
  
    p = phi_old[m]
               + time_step*.5*(dphi[m] + dphi_old[m]);

    phi[m]    = p;

    unsigned idx = (k/Size[1])%Size[0];
    unsigned idy = k%Size[1];
    unsigned idz = k/(Size[0]*Size[1]);

    const auto cp = make_cuDoubleComplex(p, 0.);    
    const auto cmx = cuCmul(com_x_table[idx],cp);
    const auto cmy = cuCmul(com_y_table[idy],cp);
    const auto cmz = cuCmul(com_z_table[idz],cp);
    atomicAdd(&com_x[n].x, cmx.x); 
    atomicAdd(&com_x[n].y, cmx.y); 
    atomicAdd(&com_y[n].x, cmy.x); 
    atomicAdd(&com_y[n].y, cmy.y); 
    atomicAdd(&com_z[n].x, cmz.x); 
    atomicAdd(&com_z[n].y, cmz.y); 

    
    atomicAdd(&vol[n],p*p);//-->vol[n]
    
  sum_one[k] = 0;
  sum_two[k] = 0;
  field_press[k] = 0;
  field_velx[k] = 0;
  field_vely[k] = 0;
  field_velz[k] = 0;
  /*
  field_polx[k] = 0;
  field_poly[k] = 0;
  field_polz[k] = 0;
  */
  
  
}

/*
__global__
void cuUpdateAtCell(				double *phi, 
					  	double *phi_dx,
					  	double *phi_dy,
					  	double *phi_dz,
					  	double *dphi,
					  	double *dphi_old,
					  	double *phi_old,
					  	double *sum_one, 
					  	double *sum_two,
					  	double *field_press,
					  	double *field_velx,
					  	double *field_vely,
					  	double *field_velz,
					  	double *vol,
					  	double *V,
					  	cuDoubleComplex *com_x,
					  	cuDoubleComplex *com_y,
					  	cuDoubleComplex *com_z,
					  	double *theta_pol,
					  	double *theta_pol_old,
					  	double time_step,
					  	cuDoubleComplex *com_x_table,
					  	cuDoubleComplex *com_y_table,
					  	cuDoubleComplex *com_z_table,
					  	double alpha,
					  	double xi,
				  		vec<double,3> *Fpressure,
				  		vec<double,3> *Fpol,
				  		vec<double,3> *velocity,
				  		vec<double,3> *polarization,
				  		vec<double,3> *com,
				  		double *delta_theta_pol,
					  	coord patch_size,
					  	coord *patch_min,
					  	coord *patch_max,
					  	coord patch_margin,
					  	coord Size,
					  	coord *offset,
					  	int n_total,
					  	unsigned patch_N,
					  	double Spol,
					  	double Dpol,
					  	double Kpol,
					  	double Jpol,
					  	unsigned N,
					  	curandState *rand_states,
					  	bool store,int cuCheck)
{

	
	// build indices with cuda!!
	const int m = blockIdx.x*blockDim.x + threadIdx.x;
	if(m>=n_total) return;
	unsigned n = static_cast<unsigned>(m)/patch_N;
	unsigned q = static_cast<unsigned>(m)%patch_N;
	
	const coord qpos = { (q/patch_size[1])%patch_size[0] , q%patch_size[1]  , q/( patch_size[0]*patch_size[1] ) };
    	const coord dpos = ( (qpos + offset[n])%patch_size + patch_min[n] )%Size;
    	const auto k = dpos[1] + Size[1]*dpos[0] + Size[0]*Size[1]*dpos[2];
	double p = phi[m];

  if(q==0){
	// cuUpdateStructureTensorAtNode<<<blocksPerGrid, threadsPerBlock>>>(n);
	// -----------------------------------------------------------------------------
	// UpdatePolarization(n, store);
	// -----------------------------------------------------------------------------
	// euler-marijuana update
	if(store){
	theta_pol_old[n] = theta_pol[n] + sqrt(time_step)*Dpol*curand_normal(&rand_states[n]);
	}
	vec<double, 3> ff = {0, 0, 0};
	ff = Fpressure[n];
	theta_pol[n] = theta_pol_old[n] - time_step*(
	+ Kpol*delta_theta_pol[n]
	+ Jpol*ff.abs() * atan2( 
	sqrt(pow( (ff[1]*polarization[n][0]-ff[0]*polarization[n][1]) ,2) + pow( (ff[2]*polarization[n][0]-ff[0]*polarization[n][2]) ,2) + pow( (ff[2]*polarization[n][1]-ff[1]*polarization[n][2]) ,2) ),
	ff[0]*polarization[n][0]+ff[1]*polarization[n][1]+ff[2]*polarization[n][2]                               
		               ));            
	polarization[n] = { Spol*cos(theta_pol[n]), Spol*sin(theta_pol[n]) };
	// -----------------------------------------------------------------------------
	// UpdateNematic(n, store);
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// ComputeCoM(n);
	// -----------------------------------------------------------------------------
	const auto mx = cuCarg(cuCdiv(com_x[n], make_cuDoubleComplex(static_cast<double>(N), 0.0))) + Pi;
	const auto my = cuCarg(cuCdiv(com_y[n], make_cuDoubleComplex(static_cast<double>(N), 0.0))) + Pi;
	const auto mz = cuCarg(cuCdiv(com_z[n], make_cuDoubleComplex(static_cast<double>(N), 0.0))) + Pi;
	// printf("bef: %u %g %g %g %i\n",n,com[n][0],com[n][1],com[n][2],cuCheck);
	// printf("%u %g %g %g\n",n,mx,my,mz);
	com[n] = { mx/2./Pi*Size[0], my/2./Pi*Size[1] , mz/2./Pi*Size[2] };
	// printf("aft: %u %g %g %g %i\n",n,com[n][0],com[n][1],com[n][2],cuCheck);
	// -----------------------------------------------------------------------------
	// UpdatePatch(n);
	// -----------------------------------------------------------------------------
	const coord com_grd { unsigned(round(com[n][0])), unsigned(round(com[n][1])), unsigned(round(com[n][2])) };
	const coord new_min = ( com_grd + Size - patch_margin ) % Size;
	const coord new_max = ( com_grd + patch_margin - coord {1u, 1u} ) % Size;
	coord displacement  = ( Size + new_min - patch_min[n] ) % Size;
	if(displacement[0]==Size[0]-1u) displacement[0] = patch_size[0]-1u;
	if(displacement[1]==Size[1]-1u) displacement[1] = patch_size[1]-1u;
	if(displacement[2]==Size[2]-1u) displacement[2] = patch_size[2]-1u;  
	// update offset and patch location
	offset[n]    = ( offset[n] + patch_size - displacement ) % patch_size;
	patch_min[n] = new_min;
	patch_max[n] = new_max;
	
    }
    
}
*/

__global__
void cuUpdateAtCell(			  	cuDoubleComplex *com_x,
					  	cuDoubleComplex *com_y,
					  	cuDoubleComplex *com_z,
					  	double *theta_pol,
					  	double *theta_pol_old,
					  	double time_step,
					  	cuDoubleComplex *com_x_table,
					  	cuDoubleComplex *com_y_table,
					  	cuDoubleComplex *com_z_table,
					  	double alpha,
					  	double xi,
				  		vec<double,3> *Fpressure,
				  		vec<double,3> *polarization,
				  		vec<double,3> *com,
				  		double *delta_theta_pol,
					  	coord patch_size,
					  	coord *patch_min,
					  	coord *patch_max,
					  	coord patch_margin,
					  	coord Size,
					  	coord *offset,
					  	int n_total,
					  	double Spol,
					  	double Dpol,
					  	double Kpol,
					  	double Jpol,
					  	unsigned N,
					  	curandState *rand_states,
					  	bool store)
{

	
	// build indices with cuda!!
	const int m = blockIdx.x*blockDim.x + threadIdx.x;
	if(m>=n_total) return;

	// cuUpdateStructureTensorAtNode<<<blocksPerGrid, threadsPerBlock>>>(n);
	// -----------------------------------------------------------------------------
	// UpdatePolarization(n, store);
	// -----------------------------------------------------------------------------
	// euler-marijuana update
	if(store){
	theta_pol_old[m] = theta_pol[m] + sqrt(time_step)*Dpol*curand_normal(&rand_states[m]);
	}
	vec<double, 3> ff = {0, 0, 0};
	ff = Fpressure[m];
	theta_pol[m] = theta_pol_old[m] - time_step*(
	+ Kpol*delta_theta_pol[m]
	+ Jpol*ff.abs() * atan2( 
	sqrt(pow( (ff[1]*polarization[m][0]-ff[0]*polarization[m][1]) ,2) + pow( (ff[2]*polarization[m][0]-ff[0]*polarization[m][2]) ,2) + pow( (ff[2]*polarization[m][1]-ff[1]*polarization[m][2]) ,2) ),
	ff[0]*polarization[m][0]+ff[1]*polarization[m][1]+ff[2]*polarization[m][2]                               
		               ));            
	polarization[m] = { Spol*cos(theta_pol[m]), Spol*sin(theta_pol[m]) };
	// -----------------------------------------------------------------------------
	// UpdateNematic(n, store);
	// -----------------------------------------------------------------------------
	// -----------------------------------------------------------------------------
	// ComputeCoM(n);
	// -----------------------------------------------------------------------------
	const auto mx = cuCarg(cuCdiv(com_x[m], make_cuDoubleComplex(static_cast<double>(N), 0.0))) + Pi;
	const auto my = cuCarg(cuCdiv(com_y[m], make_cuDoubleComplex(static_cast<double>(N), 0.0))) + Pi;
	const auto mz = cuCarg(cuCdiv(com_z[m], make_cuDoubleComplex(static_cast<double>(N), 0.0))) + Pi;
	com[m] = { mx/2./Pi*Size[0], my/2./Pi*Size[1] , mz/2./Pi*Size[2] };
	// -----------------------------------------------------------------------------
	// UpdatePatch(n);
	// -----------------------------------------------------------------------------
	const coord com_grd { unsigned(round(com[m][0])), unsigned(round(com[m][1])), unsigned(round(com[m][2])) };
	const coord new_min = ( com_grd + Size - patch_margin ) % Size;
	const coord new_max = ( com_grd + patch_margin - coord {1u, 1u} ) % Size;
	coord displacement  = ( Size + new_min - patch_min[m] ) % Size;
	if(displacement[0]==Size[0]-1u) displacement[0] = patch_size[0]-1u;
	if(displacement[1]==Size[1]-1u) displacement[1] = patch_size[1]-1u;
	if(displacement[2]==Size[2]-1u) displacement[2] = patch_size[2]-1u;  
	// -----------------------------------------------------------------------------
	// Update offset and patch location(n)
	// -----------------------------------------------------------------------------
	offset[m]    = ( offset[m] + patch_size - displacement ) % patch_size;
	patch_min[m] = new_min;
	patch_max[m] = new_max;
	
}
    


__host__ void Model::Update(bool store, unsigned start)
{
	
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "model launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    
    cuUpdateSumsAtNode<<<n_blocks, n_threads>>>(d_phi,
						      d_sum_one,
				                    d_sum_two,
                              		      d_field_polx,
                              		      d_field_poly,
                                                d_field_polz,
                                                d_field_velx,
                                                d_field_vely,
                                                d_field_velz,
                              		      d_polarization,
                             		      d_velocity,
                             		      patch_N,
                             		      n_total,
                             		      patch_size,
				                    d_patch_min,
				                    Size,
				                    d_offset);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuUpdateSumsAtNode launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaDeviceSynchronize();
    
    cuUpdatePotAtNode<<<n_blocks, n_threads>>>(d_neighbors,
                             d_neighbors_patch,
                             d_phi,
                             d_sum_one,
                             d_sum_two,
                             d_walls,
                             d_walls_laplace,
                             d_field_press,
                             d_V,
                             d_vol,
                             patch_size,
                             d_patch_min,
                             Size,
                             d_offset,
                             d_Fpol,
                             d_Fpressure,
                             d_vorticity,
                             d_delta_theta_pol,
                             kappa_cc,
                             mu,
                             lambda,
                             gam,
                             vimp,
                             omega_cc,
                             omega_cs,
                             kappa_cs,
                             n_total,
                             patch_N,
				 d_field_sxx,
				 d_field_sxy,
				 d_field_sxz,
				 d_field_syy,
				 d_field_syz,
				 d_field_szz,
				 xi,
				 d_field_velx,
				 d_field_vely,
				 d_field_velz);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuUpdatePotAtNode launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaDeviceSynchronize();
    
    
    cuUpdatePhysicalFieldsAtNode<<<n_blocks, n_threads>>>(d_neighbors,
                                     d_neighbors_patch,
                                     d_phi,
                                     d_phi_dx,
                                     d_phi_dy,
                                     d_phi_dz,
                                     d_field_press,
                                     d_sum_one,
                                     d_vol,
                                     d_com_x,
                                     d_com_y,
                                     d_com_z,
                                     d_field_polx,
                                     d_field_poly,
                                     d_field_polz,
                                     d_field_velx,
                                     d_field_vely,
                                     d_field_velz,
                                     d_Fpressure,
                                     d_vorticity,
                                     d_delta_theta_pol,
                                     d_polarization,
                                     patch_size,
                                     d_patch_min,
                                     Size,
                                     d_offset,
                                     n_total,
                                     patch_N,
					  d_field_sxx,
					  d_field_sxy,
					  d_field_sxz,
					  d_field_syy,
					  d_field_syz,
					  d_field_szz,
					  d_cSxx,
					  d_cSxy,
					  d_cSxz,
					  d_cSyy,
					  d_cSyz,
					  d_cSzz);
					  	

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuUpdatePhysicalFieldsAtNode launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaDeviceSynchronize();
    /*
    cuUpdatePolVel<<<n_blocks, n_threads>>>(
		    alpha,
		    xi,
		    d_Fpressure,
		    d_Fpol,
		    d_velocity,
		    d_polarization,
		    patch_size,
		    d_patch_min,
		    Size,
		    d_offset,
		    n_total,
		    patch_N
    );
    */
    
    nph_total   = static_cast<int>(nphases);
    nph_blocks  = (nph_total + ThreadsPerBlock - 1) / ThreadsPerBlock;
    nph_threads = ThreadsPerBlock;
    cuUpdatePolVel<<<nph_blocks, nph_threads>>>(
		    alpha,
		    xi,
		    d_Fpressure,
		    d_Fpol,
		    d_velocity,
		    d_polarization,
		    nphases);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuUpdatePolVel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaDeviceSynchronize();
    
    cuUpdatePhaseFieldAtNode<<<n_blocks, n_threads>>>(d_phi,
                                 d_phi_dx,
                                 d_phi_dy,
                                 d_phi_dz,
                                 d_dphi,
                                 d_dphi_old,
                                 d_phi_old,
                                 d_sum_one,
                                 d_sum_two,
                                 d_field_press,
                                 d_field_velx,
                                 d_field_vely,
                                 d_field_velz,
                                 d_vol,
                                 d_V,
                                 d_com_x,
                                 d_com_y,
                                 d_com_z,
                                 d_theta_pol,
                                 d_theta_pol_old,
                                 time_step,
                                 d_com_x_table,
                                 d_com_y_table,
                                 d_com_z_table,
                                 alpha,
                                 xi,
                                 d_Fpressure,
                                 d_Fpol,
                                 d_velocity,
                                 d_polarization,
                                 d_com,
                                 d_delta_theta_pol,
                                 patch_size,
                                 d_patch_min,
                                 d_patch_max,
                                 patch_margin,
                                 Size,
                                 d_offset,
                                 n_total,
                                 patch_N,
                                 Spol,
                                 Dpol,
                                 Kpol,
                                 Jpol,
                                 N,
                                 d_rand_states,
                                 store);
                                 /*
                                 d_field_polx,
                                 d_field_poly,
                                 d_field_polz);
                                 */

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuUpdatePhaseFieldAtNode launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaDeviceSynchronize();
    
    /*
    cuUpdateAtCell<<<n_blocks, n_threads>>>(d_phi,
                                 d_phi_dx,
                                 d_phi_dy,
                                 d_phi_dz,
                                 d_dphi,
                                 d_dphi_old,
                                 d_phi_old,
                                 d_sum_one,
                                 d_sum_two,
                                 d_field_press,
                                 d_field_velx,
                                 d_field_vely,
                                 d_field_velz,
                                 d_vol,
                                 d_V,
                                 d_com_x,
                                 d_com_y,
                                 d_com_z,
                                 d_theta_pol,
                                 d_theta_pol_old,
                                 time_step,
                                 d_com_x_table,
                                 d_com_y_table,
                                 d_com_z_table,
                                 alpha,
                                 xi,
                                 d_Fpressure,
                                 d_Fpol,
                                 d_velocity,
                                 d_polarization,
                                 d_com,
                                 d_delta_theta_pol,
                                 patch_size,
                                 d_patch_min,
                                 d_patch_max,
                                 patch_margin,
                                 Size,
                                 d_offset,
                                 n_total,
                                 patch_N,
                                 Spol,
                                 Dpol,
                                 Kpol,
                                 Jpol,
                                 N,
                                 d_rand_states,
                                 store,cuCheck);
    */

	    nph_total   = static_cast<int>(nphases);
	    nph_blocks  = (nph_total + ThreadsPerBlock - 1) / ThreadsPerBlock;
	    nph_threads = ThreadsPerBlock;			  			  	
           cuUpdateAtCell<<<nph_blocks, nph_threads>>>(
                                 d_com_x,
                                 d_com_y,
                                 d_com_z,
                                 d_theta_pol,
                                 d_theta_pol_old,
                                 time_step,
                                 d_com_x_table,
                                 d_com_y_table,
                                 d_com_z_table,
                                 alpha,
                                 xi,
                                 d_Fpressure,
                                 d_polarization,
                                 d_com,
                                 d_delta_theta_pol,
                                 patch_size,
                                 d_patch_min,
                                 d_patch_max,
                                 patch_margin,
                                 Size,
                                 d_offset,
                                 nphases,
                                 Spol,
                                 Dpol,
                                 Kpol,
                                 Jpol,
                                 N,
                                 d_rand_states,
                                 store);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuUpdateAtCell launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaDeviceSynchronize();
    
}




















/*
__global__
void Model::cuUpdateStructureTensorAtNode(unsigned n)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned q = z * nx * ny + y * nx + x;

	const auto  dx = phi_dx[n][q];
	const auto  dy = phi_dy[n][q];
	const auto  dz = phi_dz[n][q];
	const auto  p  = phi[n][q];

  S00[n] += -(dx*dx - (1./3.)*(dx*dx+dy*dy+dz*dz));
  S01[n] += -dx*dy;
  S02[n] += -dx*dz;
  S12[n] += -dy*dz;
  S11[n] += -(dy*dy - (1./3.)*(dx*dx+dy*dy+dz*dz));
  S22[n] += -(dz*dz - (1./3.)*(dx*dx+dy*dy+dz*dz));
}
*/


/*
void Model::UpdateNematic(unsigned n, bool store)
{
  // euler-marijuana update
  if(store)
    theta_nem_old[n] = theta_nem[n] + sqrt_time_step*Dnem*random_normal();

  double F00 = 0, F01 = 0;
  switch(align_nematic_to)
  {
    case 0:
    {
      const auto ff = velocity[n];
      F00 =   ff[0]*ff[0]
            - ff[1]*ff[1];
      F01 = 2*ff[0]*ff[1];
      break;
    }
    case 1:
    {
      const auto ff = Fpressure[n];
      F00 =   ff[0]*ff[0]
            - ff[1]*ff[1];
      F01 = 2*ff[0]*ff[1];
      break;
    }
    case 2:
      F00 = S00[n];
      F01 = S01[n];
      break;
  }
  const auto strength = pow(F01*F01 + F00*F00, 0.25);

  theta_nem[n] = theta_nem_old[n] - time_step*(
      + Knem*tau[n]
      + Jnem*strength*atan2( F00*Q01[n]-F01*Q00[n], F00*Q00[n]+F01*Q01[n]) );
      // + Wnem*vorticity[n]; need to update for 3D model 
  
  Q00[n] = Snem*cos(2*theta_nem[n]);
  Q01[n] = Snem*sin(2*theta_nem[n]);
}
*/




