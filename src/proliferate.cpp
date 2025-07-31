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
  cout<<"n :"<<n<<" "<<divisiontthresh[i]<<" "<<timer[i]<<" "<<stored_tmean[i]<<endl;
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
  
  stored_gam.resize(new_nphases,0.);
  stored_omega_cc.resize(new_nphases,0.);
  stored_omega_cs.resize(new_nphases,0.);
  stored_alpha.resize(new_nphases,0.);
  stored_dpol.resize(new_nphases,0.);
  
  timer.resize(new_nphases,0.);
  divisiontthresh.resize(new_nphases,0.);  
  stored_tmean.resize(new_nphases,0.);
  
  /*  
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

void Model::DivideCell(unsigned n, unsigned idx, double division_orientation, double cellProp){

  double px = com[n][0];
  double py = com[n][1];
  double pz = com[n][2];
  
  stored_gam[idx] = stored_gam[n];
  stored_gam[idx-1] = stored_gam[n];
  
  stored_omega_cc[idx] = cellProp;
  stored_omega_cc[idx-1] = cellProp;

  stored_omega_cs[idx] = stored_omega_cs[n];
  stored_omega_cs[idx-1] = stored_omega_cs[n];
  
  stored_alpha[idx] = stored_alpha[n];
  stored_alpha[idx-1] = stored_alpha[n];
  
  stored_dpol[idx] = stored_dpol[n];
  stored_dpol[idx-1] = stored_dpol[n];
  
  patch_min[idx] = patch_min[n];
  patch_min[idx-1] = patch_min[n];
  patch_max[idx] = patch_max[n];
  patch_max[idx-1] = patch_max[n];
  
  offset[idx] = offset[n];
  offset[idx-1] = offset[n];
  
  timer[idx] = 0.;
  timer[idx-1] = 0.;

	stored_tmean[idx] = relax_time + random_exponential(1./prolif_freq_mean);// mean = 1/lambda
	divisiontthresh[idx] = 0.;

	stored_tmean[idx-1] = relax_time + random_exponential(1./prolif_freq_mean);// mean = 1/lambda
	divisiontthresh[idx-1] = 0.;

  
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
  double epsilon = 75.;
  
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
  //com[n] = {com[nbirth][0],comy,com[nbirth][2]};
  com[n] = {comx,comy,comz};
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
	
	stored_gam.erase(stored_gam.begin()+i);
	stored_omega_cc.erase(stored_omega_cc.begin()+i);
	stored_omega_cs.erase(stored_omega_cs.begin()+i);
	stored_alpha.erase(stored_alpha.begin()+i);
	stored_dpol.erase(stored_dpol.begin()+i);
	timer.erase(timer.begin()+i);
	divisiontthresh.erase(divisiontthresh.begin()+i);
	stored_tmean.erase(stored_tmean.begin()+i);
	

}


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



double Model::UpdateOU(double tcurrent, double tmean, double tcorr, double sigma, const double dt){
double dW = std::sqrt(dt) * random_normal(1.);
return tcurrent - ((tcurrent-tmean) / tcorr) * dt + sigma * dW;
}



void Model::initDivisionOU(unsigned n, unsigned i, double division_orientation, unsigned t, bool mutate){
	double relt = static_cast<double>(t) / (nsubsteps * ninfo);
	BirthCellMemories(nphases_index.size() + 3);
	
	int cellGen = cellHist[n].generation + 1;
	double cellProp = stored_omega_cc[i];
	if (mutate){
	cellProp = cellProp + cellProp * mutation_strength;
	if (cellProp > max_prop_val) cellProp = max_prop_val;
	if (cellProp < min_prop_val) cellProp = min_prop_val;
	} 
	nphases_index_head = nphases_index_head + 1;
	nphases_index.push_back(nphases_index_head);

	nphases_index_head = nphases_index_head + 1;
	nphases_index.push_back(nphases_index_head);

	unsigned curr_idx = nphases_index.size() - 1;
	cellLineage(/*cell_id=*/curr_idx,/*parent_id=*/n,/*birth_time=*/relt,/*death_time=*/-1,/*physicalprop=*/cellProp,/*generation=*/cellGen);
	cellLineage(/*cell_id=*/curr_idx-1,/*parent_id=*/n,/*birth_time=*/relt,/*death_time=*/-1,/*physicalprop=*/cellProp,/*generation=*/cellGen);
	DivideCell(i,curr_idx,division_orientation,cellProp);
	BirthCell(curr_idx);
	BirthCell(curr_idx-1);
	ComputeBirthCellCOM(curr_idx,i);
	ComputeBirthCellCOM(curr_idx-1,i);
	cellLineage(/*cell_id=*/n,/*parent_id=*/-1,/*birth_time=*/-1,/*death_time=*/relt,/*physicalprop=*/cellProp,/*generation=*/cellGen);
	KillCell(n,i);
}



std::vector<double> Model::compute_eigen(double sxx, double sxy, double syy){

        double trace = sxx + syy;
        double delta = std::sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy);
        double eig1 = 0.5 * (trace + delta);
        double eig2 = 0.5 * (trace - delta);

        double eigvalmax, eigvalmin;
        if (eig1 >= eig2) {
            eigvalmax = eig1;
            eigvalmin = eig2;
        } else {
            eigvalmax = eig2;
            eigvalmin = eig1;
        }

        // Compute eigenvectors.
        double epsilon = std::numeric_limits<double>::epsilon();
        double v1_x, v1_y, v2_x, v2_y;
        if (std::abs(sxy) > epsilon) {
            v1_x = eigvalmax - syy;
            v1_y = sxy;
            v2_x = eigvalmin - syy;
            v2_y = sxy;
        } else {
            // Diagonal matrix: choose standard unit vectors.
            if (sxx >= syy) {
                v1_x = 1.0; v1_y = 0.0;
                v2_x = 0.0; v2_y = 1.0;
            } else {
                v1_x = 0.0; v1_y = 1.0;
                v2_x = 1.0; v2_y = 0.0;
            }
        }
        // Normalize eigenvectors.
        double norm1 = std::sqrt(v1_x * v1_x + v1_y * v1_y);
        double norm2 = std::sqrt(v2_x * v2_x + v2_y * v2_y);
        if (norm1 > epsilon) {
            v1_x /= norm1;
            v1_y /= norm1;
        }
        if (norm2 > epsilon) {
            v2_x /= norm2;
            v2_y /= norm2;
        }

	return {eigvalmax, eigvalmin, v1_x, v1_y, v2_x, v2_y};
}


void Model::cellLineage(int cell_id, int parent_id, double birth_t, double death_t, double prop, int gen)
{
    // 1) Check if cell_id already exists
    auto it = cellHist.find(cell_id);
    if (it != cellHist.end()) {
        // -- EXISTING CELL --
        // Only update fields that make sense for your scenario.
        // For example, if death_t != -1.0, update the death_time:
        if (death_t != -1.0) {
            it->second.death_time = death_t;
        }

        // If you also want to overwrite parent, generation, etc.:
        // it->second.parent     = parent_id;  
        // it->second.birth_time = birth_t;     
        // it->second.generation = gen;
        // (Or use sentinel logic to skip fields.)
    }
    else {
        // -- NEW CELL --
        cellInfo info;
        info.birth_time = birth_t;
        info.death_time = death_t;      // -1.0 => alive
        info.parent     = parent_id;    // -1 => no parent
        info.generation = gen;
        info.physicalprop = prop;


        // Insert in the global map
        cellHist[cell_id] = info;
    }
}

/*
void Model::stress_criterionOU(unsigned pcellIndex, bool &mutate, double &angle) {

    double cellStress = (cSxx[pcellIndex] + cSyy[pcellIndex] + cSzz[pcellIndex]) / 3.0;
    mutate = (cellStress > avgStress);

    const double sxx = cSxx[pcellIndex];
    const double sxy = cSxy[pcellIndex];
    const double syy = cSyy[pcellIndex];
    const auto eigenResults = compute_eigen(sxx, sxy, syy);
    angle = std::atan2(eigenResults[3], eigenResults[2]);
}
*/

double Model::compute_percentile(double p, vector<double>& fdata) {
    if(fdata.empty()) return 0.0;
    sort(fdata.begin(), fdata.end());
    double index = (p / 100.0) * (fdata.size() - 1);
    if(index == floor(index)) {
        return fdata[static_cast<int>(index)];
    }
    int lower_idx = static_cast<int>(floor(index));
    int upper_idx = lower_idx + 1;
    if(upper_idx >= fdata.size()) {
        return fdata.back();
    }
    double fraction = index - lower_idx;
    return fdata[lower_idx] + fraction * (fdata[upper_idx] - fdata[lower_idx]);
}


void Model::proliferate_stress_based(unsigned t) {

	double pcompglobal = 0.;
	double ptensglobal = 0.;
	double wcompglobal = 0.;
	double wtensglobal = 0.;
	vector<double> pcompdata;
	vector<double> ptensdata;
	vector<unsigned> detached;
	for(unsigned i=0; i<nphases_index.size(); ++i){
	if ((com[i][2] - wall_thickness) > 4.*R) {
	  detached.push_back(i);
	}
       for(unsigned q=0; q<patch_N; ++q){
    	const auto  k  = GetIndexFromPatch(i, q);
    	double press = (1./3.) * (field_sxx[k] + field_syy[k] + field_szz[k]);
    	if (press <= 0.){
    	pcompglobal += phi[i][q] * press;
    	wcompglobal += phi[i][q];
    	pcompdata.push_back(press);
    	}
    	if (press > 0.){
    	ptensglobal += phi[i][q] * press;
    	wtensglobal += phi[i][q];
    	ptensdata.push_back(press);
    	}
	}
	}
	ptensglobal /= wtensglobal;
	pcompglobal /= wcompglobal;
	sort(pcompdata.begin(), pcompdata.end());
	sort(ptensdata.begin(), ptensdata.end());
	
	// double p60 = compute_percentile(60,fdata);
	// double p70 = compute_percentile(70,fdata);
	// double p80 = compute_percentile(80,fdata);
	// double p90 = compute_percentile(90,fdata);
	
	/*
	if (t > prolif_start){
	cout<<t<<" "<<prolif_start<<endl;
	}
	*/

	unsigned i = 0;
	while (i < nphases_index.size()) {
		unsigned n = nphases_index[i];
		Write_OU(t, i);
		timer[i] += 1;
		divisiontthresh[i] = UpdateOU(divisiontthresh[i], stored_tmean[i], tcorr, sigma, 1.);
		
		double plocal = 0.;
		double sxxlocal = 0.;
		double sxylocal = 0.;
		double syylocal = 0.;
		double wlocal = 0.;
       	for(unsigned q=0; q<patch_N; ++q){
    		const auto  k  = GetIndexFromPatch(i, q);
       	plocal += phi[i][q] * (1./3.)*(field_sxx[k]+field_syy[k]+field_szz[k]);
       	sxxlocal += phi[i][q] * field_sxx[k];
       	sxylocal += phi[i][q] * field_sxy[k];
       	syylocal += phi[i][q] * field_syy[k];
       	wlocal += phi[i][q];
		}
		
		plocal /= wlocal;
		sxxlocal /= wlocal;
		syylocal /= wlocal;
		sxylocal /= wlocal;
		
		// stress_based_prolif_criterion =  (pglobal * (plocal - pglobal)) > 0;
              if (proliferate_bool && (t > prolif_start) && nphases_index.size() < nphases_max && timer[i] >= divisiontthresh[i] && (com[i][2]-wall_thickness) < 3.*R) {
		cout<<"dividing cell "<<i<<" "<<n<<" "<<timer[i]<<" "<<divisiontthresh[i]<<endl;
		
		bool mutate = false;

		if (plocal <= 0. && plocal < pcompglobal){
		mutate = true;
		}
		if (plocal > 0. && plocal > ptensglobal){
		mutate = true;
		}
    		const auto eigenResults = compute_eigen(sxxlocal, sxylocal, syylocal);
    		double angle = std::atan2(eigenResults[3], eigenResults[2]);
    
		GetFromDevice();
		FreeDeviceMemoryCellBirth();
		initDivisionOU(n, i, angle, t, mutate);

		while (!detached.empty()) {
		unsigned j = detached.back();
		detached.pop_back();
		cout<<"removing :"<<j<<" "<<nphases_index[j]<<endl;
		KillCell(nphases_index[j], j);
		}
		print_new_cell_props();
		nphases = nphases_index.size();
		Write_divAngle(t, n, i, mutate, angle,plocal,pcompglobal,ptensglobal);
		AllocDeviceMemoryCellBirth();
		PutToDevice();
		cout << "proliferation complete; current number at " << nphases << endl;
		}
		i++; // Move to the next index; the condition is re-evaluated based on the current size.
	}
}




   /* 
void Model::proliferate(unsigned t) {


	 
	 vector<unsigned> detached;
	for(unsigned i=0; i<nphases_index.size(); ++i){
	     if ((com[i][2] - wall_thickness) > 30) {
		  detached.push_back(i);
	     }
	     }
	 
	 // print_new_cell_props();
	 unsigned i = 0;
        while (i < nphases_index.size()) {
            unsigned n = nphases_index[i];
            Write_OU(t, i);
            timer[i] += 1;
            divisiontthresh[i] = UpdateOU(divisiontthresh[i], stored_tmean[i], tcorr, sigma, 1.);


            if (proliferate_bool && (t > prolif_start) && nphases_index.size() < nphases_max && timer[i] >= divisiontthresh[i] and (com[i][2]-wall_thickness) < 2*R) {
                bool mutate = false;
                double angle = 0.;
                stress_criterionOU(i, mutate, angle);
                GetFromDevice();
                FreeDeviceMemoryCellBirth();
                initDivisionOU(n, i, angle, t, mutate);
    
    while (!detached.empty()) {
        unsigned j = detached.back();
        detached.pop_back();
        cout<<"removing :"<<j<<" "<<nphases_index[j]<<endl;
        KillCell(nphases_index[j], j);
    }
    
                //print_new_cell_props();
                nphases = nphases_index.size();
                Write_divAngle(t, n, i, mutate, angle);
                // nphases_index_head = nphases;
                AllocDeviceMemoryCellBirth();
                PutToDevice();
                cout << "proliferation complete; current number at " << nphases << endl;
            }
            i++; // Move to the next index; the condition is re-evaluated based on the current size.
        }
    }
*/


void Model::write_cellHist_binary(const std::string &filename,
                                  unsigned currentTime,
                                  const std::map<int, cellInfo> &hist)
{
    std::ofstream out(filename, std::ios::binary | std::ios::app);
    if (!out) {
        std::cerr << "Could not open file " << filename << " for binary write.\n";
        return;
    }

    double timeVal = static_cast<double>(currentTime);
    out.write(reinterpret_cast<const char*>(&timeVal), sizeof(timeVal));

    int nCells = static_cast<int>(hist.size());
    out.write(reinterpret_cast<const char*>(&nCells), sizeof(nCells));

    for (auto &kv : hist) {
        int cellID = kv.first;
        const cellInfo &ci = kv.second;

        out.write(reinterpret_cast<const char*>(&cellID), sizeof(cellID));
        out.write(reinterpret_cast<const char*>(&ci.birth_time),    sizeof(ci.birth_time));
        out.write(reinterpret_cast<const char*>(&ci.death_time),    sizeof(ci.death_time));
        out.write(reinterpret_cast<const char*>(&ci.parent),        sizeof(ci.parent));
        out.write(reinterpret_cast<const char*>(&ci.physicalprop),  sizeof(ci.physicalprop));
        out.write(reinterpret_cast<const char*>(&ci.generation),    sizeof(ci.generation));
    }

    out.close();
}














