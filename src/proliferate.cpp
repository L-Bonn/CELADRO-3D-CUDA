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
 
#include "header.hpp"
#include "model.hpp"
#include "derivatives.hpp"

using namespace std;

void Model::print_new_cell_props(){
  
  unsigned n;
  for(unsigned i=0; i<nphases_index.size(); ++i){
  n = nphases_index[i];
  cout<<"n :"<<n<<" i: "<<i<<" prop: "<<com[i]<<" "<<phi.size()<<endl;
  }
}


void Model::BirthCellMemories(unsigned new_nphases){

  phi.resize(new_nphases, vector<double>(patch_N, 0.));
  phi_dx.resize(new_nphases, vector<double>(patch_N, 0.));
  phi_dy.resize(new_nphases, vector<double>(patch_N, 0.));
  phi_dz.resize(new_nphases, vector<double>(patch_N, 0.));
  phi_old.resize(new_nphases, vector<double>(patch_N, 0.));
  V.resize(new_nphases, vector<double>(patch_N, 0.));
  dphi.resize(new_nphases, vector<double>(patch_N, 0.));
  dphi_old.resize(new_nphases, vector<double>(patch_N, 0.));
  vol.resize(new_nphases, 0.);
  patch_min.resize(new_nphases, {0, 0, 0});
  patch_max.resize(new_nphases, Size);
  com.resize(new_nphases, {0., 0., 0.});
  com_prev.resize(new_nphases, {0., 0., 0.});
  offset.resize(new_nphases, {0u, 0u, 0u});
  cSxx.resize(new_nphases,0.);
  cSxy.resize(new_nphases,0.);
  cSxz.resize(new_nphases,0.);
  cSyy.resize(new_nphases,0.);
  cSyz.resize(new_nphases,0.);
  cSzz.resize(new_nphases,0.);
  theta_pol.resize(new_nphases, 0.);
  theta_pol_old.resize(new_nphases, 0.);
  delta_theta_pol.resize(new_nphases, 0.);
  polarization.resize(new_nphases, {0., 0., 0.});
  vorticity.resize(new_nphases,{0.,0.,0.});
  velocity.resize(new_nphases, {0., 0., 0.});
  Fpressure.resize(new_nphases, {0., 0., 0.});
  Fshape.resize(new_nphases, {0., 0., 0.});
  //Fnem.resize(new_nphases, {0., 0., 0.});
  Fpol.resize(new_nphases, {0., 0., 0.});
  com_x.resize(new_nphases, 0.);
  com_y.resize(new_nphases, 0.);
  com_z.resize(new_nphases, 0.);
  
  /*
  alphas.resize(new_nphases,0.);
  zetaS_field.resize(new_nphases,0.);
  zetaQ_field.resize(new_nphases,0.);
  gams.resize(new_nphases,0.);
  omega_ccs.resize(new_nphases,0.);
  omega_cws.resize(new_nphases,0.);
  xis.resize(new_nphases,0.);
  kappas.resize(new_nphases,0.);
  mus.resize(new_nphases,0.);
  Rs.resize(new_nphases,0.);
  V0.resize(new_nphases,0.);
  cellTypes.resize(new_nphases,0.);
  
  S00.resize(new_nphases, 0.);
  S01.resize(new_nphases, 0.);
  S02.resize(new_nphases, 0.);
  S12.resize(new_nphases, 0.);
  S11.resize(new_nphases, 0.);
  S22.resize(new_nphases, 0.);
    
  Q00.resize(new_nphases, 0.);
  Q01.resize(new_nphases, 0.);
  */

}

void Model::DivideCell(unsigned n, unsigned idx){

  double px = com[n][0];
  double py = com[n][1];
  double pz = com[n][2];
  
  /*
  gams[idx] = gams[n];
  gams[idx-1] = gams[n];
  
  zetaS_field[idx] = zetaS_field[n];
  zetaS_field[idx-1] = zetaS_field[n];

  zetaQ_field[idx] = zetaQ_field[n];
  zetaQ_field[idx-1] = zetaQ_field[n];
  
  omega_ccs[idx] = omega_ccs[n];
  omega_ccs[idx-1] = omega_ccs[n];
  
  omega_cws[idx] = omega_cws[n];
  omega_cws[idx-1] = omega_cws[n];
  
  kappas[idx] = kappas[n];
  kappas[idx-1] = kappas[n];
  
  mus[idx] = mus[n];
  mus[idx-1] = mus[n];
  
  Rs[idx] = Rs[n];
  Rs[idx-1] = Rs[n];
  
  V0[idx] = V0[n];
  V0[idx-1] = V0[n];
  
  alphas[idx] = alphas[n];
  alphas[idx-1] = alphas[n];
  
  xis[idx] = xis[n];
  xis[idx-1] = xis[n];
  
  cellTypes[idx] = cellTypes[n];
  cellTypes[idx-1] = cellTypes[n];
  */
  
  patch_min[idx] = patch_min[n];
  patch_min[idx-1] = patch_min[n];
  patch_max[idx] = patch_max[n];
  patch_max[idx-1] = patch_max[n];
  
  offset[idx] = offset[n];
  offset[idx-1] = offset[n];
  
  
  double rndir = random_uniform();
  double px1 = px + (R)*cos(rndir);
  double py1 = py + (R)*sin(rndir);
  double px2 = px + (R)*cos(rndir+M_PI);
  double py2 = py + (R)*sin(rndir+M_PI);

  for(unsigned q=0; q<patch_N; ++q){
  
  const auto      k = GetIndexFromPatch(n, q);
  const unsigned xk = GetXPosition(k);
  const unsigned yk = GetYPosition(k);
  const unsigned zk = GetZPosition(k);
  
  double pa[3] = {px1,py1,pz};
  double pb[3] = {px2,py2,pz};
  double pr[3] = {xk,yk,zk};
  double epsilon = 25.;
  
  double a[3] = {pa[0]-pb[0],pa[1]-pb[1],pa[2]-pb[2]};
  double norm = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
  a[0] = a[0] / norm;
  a[1] = a[1] / norm;
  a[2] = a[2] / norm;
  
  double b[3] ={pr[0]-(pa[0]+pb[0])/2.,pr[1]-(pa[1]+pb[1])/2.,pr[2]-(pa[2]+pb[2])/2.};
  double g = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  double chi = (0.5)*(1+tanh(g/epsilon));
  
  phi[idx][q] = phi[n][q]*chi;
  phi[idx-1][q] = phi[n][q]*(1.-chi);
  
  phi_old[idx][q] = phi[n][q]*chi;
  phi_old[idx-1][q] = phi[n][q]*(1.-chi);
  
  }
}

void Model::ComputeBirthCellCOM(unsigned n, unsigned nbirth){

  double comx, comy, comz;
  comx = comy = comz = 0.;
  int count = 0;
  for(unsigned q=0; q<patch_N; ++q){
  
  const auto      k = GetIndexFromPatch(n, q);
  const unsigned xk = GetXPosition(k);
  const unsigned yk = GetYPosition(k);
  const unsigned zk = GetZPosition(k);
  
  if(phi[n][q] > 0.){
  comx = comx + xk;
  comy = comy + yk;
  comz = comz + zk;
  count = count + 1;
  }
  
  }
  comx = comx / count;
  comy = comy / count;
  comz = comz / count;
  com[n] = {com[nbirth][0],comy,com[nbirth][2]};
}

void Model::BirthCellAtNode(unsigned n, unsigned q)
{
    
  const auto      k = GetIndexFromPatch(n, q);

  phi_old[n][q] = phi[n][q];
  com_x[n] += com_x_table[GetXPosition(k)]*phi[n][q];
  com_y[n] += com_y_table[GetYPosition(k)]*phi[n][q];
  com_z[n] += com_z_table[GetZPosition(k)]*phi[n][q];
}
 
void Model::KillCell(unsigned n, unsigned i){

	for(unsigned q=0; q<patch_N; ++q){
		const auto k = GetIndexFromPatch(i, q);
		field_press[k] = 0.;
		sum_one[k] = 0;
		sum_two[k] = 0;
		field_polx[k] = 0.;
		field_poly[k] = 0.;
		field_polz[k] = 0.;
		field_velx[k] = 0;
		field_vely[k] = 0;
		field_velz[k] = 0;
	}

	phi.erase(phi.begin() + i);
	phi_old.erase(phi_old.begin() + i);
	V.erase(V.begin()+i);
	
	phi_dx.erase(phi_dx.begin()+i);
	phi_dy.erase(phi_dy.begin()+i);
	phi_dz.erase(phi_dz.begin()+i);
	
	dphi.erase(dphi.begin()+i);
	dphi_old.erase(dphi_old.begin()+i);

	com.erase(com.begin()+i);
	polarization.erase(polarization.begin()+i);
	velocity.erase(velocity.begin()+i);
	patch_min.erase(patch_min.begin()+i);
	patch_max.erase(patch_max.begin()+i);
	offset.erase(offset.begin()+i);
	vol.erase(vol.begin()+i);
	Fpol.erase(Fpol.begin()+i);
	cSxx.erase(cSxx.begin()+i);
	cSxy.erase(cSxy.begin()+i);
	cSxz.erase(cSxz.begin()+i);
	cSyy.erase(cSyy.begin()+i);
	cSyz.erase(cSyz.begin()+i);
	cSzz.erase(cSzz.begin()+i);
	Fpressure.erase(Fpressure.begin()+i);
	vorticity.erase(vorticity.begin()+i);
	delta_theta_pol.erase(delta_theta_pol.begin()+i);
	theta_pol.erase(theta_pol.begin()+i);
	theta_pol_old.erase(theta_pol_old.begin()+i);
	com_x.erase(com_x.begin()+i);
	com_y.erase(com_y.begin()+i);
	com_z.erase(com_z.begin()+i);
	
	
	nphases_index.erase(nphases_index.begin()+i);
}
/*
void Model::KillCell(unsigned n,unsigned i){
  for(unsigned q=0; q<patch_N; ++q){
  const auto k = GetIndexFromPatch(n, q);
  
  phi[n][q] = 0.;
  phi_old[n][q] = 0.;
  V[n][q] = 0.;
  pressure[k] = 0.;

  sum_one[k] = 0;
  sum_two[k] = 0;

  //sumS00[k] = 0;
  //sumS01[k] = 0;
  //sumS02[k] = 0;
  //sumS12[k] = 0;
  //sumS11[k] = 0;
  //sumS22[k] = 0;
  
  //sumQ00[k] = 0;
  //sumQ01[k] = 0;
  
  P0[k] = 0.;
  P1[k] = 0.;
  P2[k] = 0.;
  
  U0[k] = 0;
  U1[k] = 0;
  U2[k] = 0;
  
  }
  nphases_index.erase(nphases_index.begin()+i);
}
*/


void Model::BirthCell(unsigned n)
{

  // init polarisation and nematic
  theta_pol[n] = noise*Pi*(1-2*random_real());
  polarization[n] = { Spol*cos(theta_pol[n]), Spol*sin(theta_pol[n]) };
  
  /*
  theta_nem[n] = noise*Pi*(1-2*random_real());
  Q00[n] = Snem*cos(2*theta_nem[n]);
  Q01[n] = Snem*sin(2*theta_nem[n]);
  */

}


void Model::initDivision(unsigned n, unsigned i){
	BirthCellMemories(nphases_index_head + 3);
	nphases_index_head = nphases_index_head + 1;
	//cout<<"current number of cells: "<<nphases_index_head<<endl;
	nphases_index.push_back(nphases_index_head);
	nphases_index_head = nphases_index_head + 1;
	//cout<<"current number of cells: "<<nphases_index_head<<endl;
	nphases_index.push_back(nphases_index_head);
	DivideCell(i,nphases_index_head);
	BirthCell(nphases_index_head);
	BirthCell(nphases_index_head-1);

	ComputeBirthCellCOM(nphases_index_head,i);
	ComputeBirthCellCOM(nphases_index_head-1,i);
	//print_new_cell_props();
	KillCell(n,i);
}


void Model::proliferate(unsigned t){

  /*
  unsigned n;
  for(unsigned i=0; i<nphases_index.size(); ++i){
  n = nphases_index[i];
  if( com[i][2] > (wall_thickness + 1.5 * R) ){
  KillCell(n,i);
  }
  }
  */
    
	if(proliferate_bool and t > prolif_start and remainder(tau_divide*1.,prolif_freq)==0. and nphases_index.size()*1. < nphases_max){
	       //cout<<"init nphases: "<<nphases<<endl;
		GetFromDevice();
		FreeDeviceMemoryCellBirth();
		int imax = nphases_index.size() - 1;
		unsigned i = random_int_uniform(0,imax);
		//unsigned i = 1;
		unsigned n = nphases_index[i];
		cout<<"dividing cell "<<n<<" with index "<<i<<endl;
		//print_new_cell_props();
		initDivision(n,i);
		//print_new_cell_props();
		//cout<<"initDivision was successful "<<endl;
		nphases = nphases_index.size();
		nphases_index_head = nphases - 1;
		AllocDeviceMemoryCellBirth();
		PutToDevice();
		//Write_visData(t);
		cout<<"proliferation complete; current number at "<<nphases<<endl;
		tau_divide = 0;
	}
	tau_divide = tau_divide + 1;
}
















