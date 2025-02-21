#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <set>
#include <iterator>
#include <numeric> 
#include <chrono>
#include <random>
#include <complex>
#include <algorithm>

using namespace std;

int seed, mcsteps;
double d_max_mc;

double* xcf;
double* ycf;
double* zcf;
double* accepted_moves;






double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);

}

void write_simCard(const string& config, const string& config_subAdh, double gamma, double omegacc, double omegacw, double alpha, int nsteps, int ninfo, double Lx, double Ly, double Lz, int nsubsteps, int bc, double margin, int relax_time, int nphases, double mu, double lambda, double kappa, double rad, double xi, double wallThich, double wallKappa, double SPOL, double DPOL, double JPOL,double KPOL, double kij, double zetaS, double zetaQ,double SNEM,double KNEM,double JNEM,double WNEM, int count){
        
    FILE * sortie;
    char nomfic[256];
    sprintf(nomfic, "simCard_%d.dat", count);
    
    sortie = fopen(nomfic, "w");
    fprintf(sortie, "# Sample runcard\n");
    fprintf(sortie, "config = %s\n", config.c_str() );
	fprintf(sortie, "nsteps = %d\n", nsteps);
	fprintf(sortie, "ninfo = %d\n", ninfo);
	fprintf(sortie, "LX = %g\n", Lx);
	fprintf(sortie, "LY = %g\n", Ly);
	fprintf(sortie, "LZ = %g\n", Lz);
	fprintf(sortie, "nsubsteps = %d\n", nsubsteps);
	fprintf(sortie, "bc = %d\n", bc);
	fprintf(sortie, "margin = %g\n", margin);
	fprintf(sortie, "relax-time = %d\n", relax_time);
    fprintf(sortie, "substrate-adhesion = %s\n", config_subAdh.c_str() );
	fprintf(sortie, "nphases = %d\n", nphases);
	fprintf(sortie, "gamma = %g\n", gamma);
	fprintf(sortie, "mu = %g\n", mu);
	fprintf(sortie, "lambda = %g\n", lambda);
	fprintf(sortie, "kappa = %g\n", kappa);
	fprintf(sortie, "R = %g\n", rad);
	fprintf(sortie, "xi = %g\n", xi);
	fprintf(sortie, "omega = %g\n", omegacc);
	fprintf(sortie, "wall-thickness = %g\n", wallThich);
	fprintf(sortie, "wall-kappa = %g\n", wallKappa);
	fprintf(sortie, "wall-omega = %g\n", omegacw);
	fprintf(sortie, "alpha = %g\n", alpha);
	fprintf(sortie, "kij = %g\n", kij);
	fprintf(sortie, "S-pol = %g\n", SPOL);
	fprintf(sortie, "D-pol = %g\n", DPOL);
	fprintf(sortie, "J-pol = %g\n", JPOL);
	fprintf(sortie, "K-pol = %g\n", KPOL);
	fprintf(sortie, "zetaS = %g\n", zetaS);
	fprintf(sortie, "zetaQ = %g\n", zetaQ);
	fprintf(sortie, "S-nem = %g\n", SNEM);
	fprintf(sortie, "K-nem = %g\n", KNEM);
	fprintf(sortie, "J-nem = %g\n", JNEM);
	fprintf(sortie, "W-nem = %g\n", WNEM);
		
    fclose(sortie);
     
}

void disorder_initial(double l, int np, double xcf[], double ycf[], double zcf[], int nx, int ny, int nz){
	
     double xc, yc, zc;
	int num;

	num = 0;
     for (int z = 0 ; z < nz ; z++){
     for (int y = 0 ; y < ny ; y++){
	for (int x = 0 ; x < nx ; x++){
	
     xc = round(l * x + l/2);
     yc = round(l * y + l/2);
	zc = round(l * z + l/2);
	xcf[num] = xc;
	ycf[num] = yc;
	zcf[num] = zc;
	
	num++;
	}
	}
	}
}

/*
void disorder_mc(size_t d_max, size_t rnd, double xcf[], double ycf[], double zcf[], int npart){
	
	double nxc,nyc,nzc,dmin,p1,p2,p3;
	int exp1, exp2, exp3;
	bool xlub, xllb, ylub, yllb, zlub, zllb, dcond;
	bool xgub, xglb, ygub, yglb, zgub, zglb;

	for (int i = 0 ; i < npart ; i++){

	exp1 = rand() %2;
	exp2 = rand() %2;
	exp3 = rand() %2;
		
	p1 = fRand(0,d_max);
	p2 = fRand(0,d_max);
	p3 = fRand(0,d_max);
		
    	xcf[i] = xcf[i] + pow(-1,exp1) * p1;
    	ycf[i] = ycf[i] + pow(-1,exp2) * p2;
	zcf[i] = zcf[i] + pow(-1,exp3) * p3;
		
	}
}
*/

void init_square_lattice(double l, int np, double xcf[], double ycf[], double zcf[], int nx, int ny, int nz){
	
     double xc, yc, zc;
	int num;

	num = 0;
     for (int z = 0 ; z < nz ; z++){
     for (int y = 0 ; y < ny ; y++){
	for (int x = 0 ; x < nx ; x++){
	
     xc = round(l * x + l/2);
     yc = round(l * y + l/2);
	zc = round(l * z + l/2);
	xcf[num] = xc;
	ycf[num] = yc;
	zcf[num] = zc;
	
	num++;
	}
	}
	}
}


void init_triangular_lattice(int nx, int ny, double domainWidth, double domainHeight, double zcoor, double xcf[], double ycf[], double zcf[], double R0) {
  //  double dx = domainWidth / (nx - 1); // Horizontal spacing
  //  double dy = domainHeight / (ny - 1); // Vertical spacing
    double dx = domainWidth / (nx); // Horizontal spacing
    double dy = domainHeight / (ny); // Vertical spacing
    
    double dx2 = dx * 0.75; // Horizontal spacing between alternate rows
    int num = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double x = i * dx;
            if (j % 2 == 1) {
                x += dx2;
            }
            double y = j * dy;
            xcf[num] = x;
            ycf[num] = y + R0;
            zcf[num] = zcoor;
            num++;
        }
    }
    cout<<"num :"<<num<<endl;
}

void disorder_mc(size_t d_max, size_t rnd, double xcf[], double ycf[], double zcf[], int npart){
	
	double nxc,nyc,nzc,dmin,p1,p2,p3;
	int exp1, exp2, exp3;
	bool xlub, xllb, ylub, yllb, zlub, zllb, dcond;
	bool xgub, xglb, ygub, yglb, zgub, zglb;

	for (int i = 0 ; i < npart ; i++){

	exp1 = rand() %2;
	exp2 = rand() %2;
	exp3 = rand() %2;
		
	p1 = fRand(0,d_max);
	p2 = fRand(0,d_max);
	p3 = fRand(0,d_max);
		
    	xcf[i] = xcf[i] + pow(-1,exp1) * p1;
    	ycf[i] = ycf[i] + pow(-1,exp2) * p2;
	zcf[i] = zcf[i] + pow(-1,exp3) * p3;
		
	}
}



void write_lattice(const string& _name,int np, double xcf[], double ycf[], double zcf[] , double zcoor){

	double xc,yc,zc,r;
   const char * c = _name.c_str();
   FILE * sortie;
   sortie = fopen(c, "w+");
    	
	for(int j = 0 ; j < np ; j++){

    	xc = xcf[j];
	yc = ycf[j];
	zc = zcf[j];

	fprintf(sortie,"%g %g %g\n", xc,yc,zcoor);

	}
 	fclose(sortie);
}
 	

double find_min(vector<double> &vect){

	double small = vect[0];
	for (size_t i = 0 ; i < vect.size() ; i++){
	if(vect[i] < small) small = vect[i];
	}
	return small;
}



double compute_all_dist(int index, double ix, double iy, double iz, double txc[], double tyc[], double tzc[], int npart){

	vector<double> dist;

	for (int j = 0 ; j < npart ; j++){
	if(index != j){
	dist.push_back( sqrt(pow( ( txc[j]-ix ),2) + pow( ( tyc[j]-iy ),2) + pow( ( tzc[j]-iz ),2) ) );
	}
	}

	double dmin = find_min(dist);
	dist.clear();
	return dmin;
}















void write_posfile_mix_perc(int np, double xcf[], double ycf[], double zcf[] , double zcoor, double na, double zetas1, double zetas2, double zetaQ1, double zetaQ2, double gam1, double gam2, double omega1,double omega2, double omega_wall1,double omega_wall2, double kappa, double mu, double alpha, double xi, double R, int count){

	double xc,yc,zc,r;
	FILE * sortie;
	char nomfic[256];
	sprintf(nomfic, "input_str_%d.dat", count);
    	sortie = fopen(nomfic, "w+");
    	
	for(int j = 0 ; j < np ; j++){

    	xc = xcf[j];
	yc = ycf[j];
	zc = zcf[j];

	if ( j < na)  {
	fprintf(sortie,"%g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", 1.,xc,yc,zcoor,zetas1,zetaQ1,gam1,omega1,omega_wall1,kappa,mu,alpha,xi,R);
	}
	else{
	fprintf(sortie,"%g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", 2.,xc,yc,zcoor,zetas2,zetaQ2,gam2,omega2,omega_wall2,kappa,mu,alpha,xi,R);
	}
	
	}
 	fclose(sortie);
}

void write_summary( int count, double gam1, double gam2, double zetas1, double zetas2, double omegacc1,double omegacc2, double omegacw1,double omegacw2, double alpha, double xi, double zetaQ){

 	FILE * sortie; 
	sortie = fopen("simulation_parameter_summary.dat","a");
	fprintf(sortie,"%i %g %g %g %g %g %g %g %g %g %g %g\n",count,gam1,gam2,zetas1,zetas2,omegacc1,omegacc2,omegacw1,omegacw2,alpha,xi,zetaQ);
	fclose(sortie);
}

void  Export(const string& _name, double tmp[], int tot){
	
   const char * c = _name.c_str();
   FILE * sortie;
   sortie = fopen(c, "w+");	
   for (int k = 0 ; k < (tot) ; k++) fprintf(sortie,"%g\n",tmp[k]);
}


double compute_mean(double tmp[], int N)
{
	double m1 = 0.;
	for (int i = 0 ; i < N ; i++){
	m1 += tmp[i];
	}
	m1 = m1/N;
	return m1;	
}










/*
def ipx(x,y,z,Lx,Ly): 
	return (y + Ly * x + Lx*Ly*z);
*/

int ipx(int x, int y, int z, int xsize, int ysize){
	return (y+ysize*x+xsize*ysize*z);
}




// coare_grain_stress_field( LXn, LYn, LZn, fxn, fyn, fzn, sigxx, sigxy, sigxz, sigyy, sigyz, sigzz );
void coare_grain_stress_field( int LX, int LY, int LZ, double fx[], double fy[], double fz[], double sigxx[], double sigxy[], double sigxz[], double sigyy[], double sigyz[], double sigzz[] ){

	int idx, idn;
	
	int nvx = LX-1;
	int nvy = LY-1;
	int nvz = LZ-1;
			
	int nsiteX = 2;
	int nsiteY = 2;
	int nsiteZ = 2;

	double* LengthX;
	double* LengthY;
	double* LengthZ;
	
	LengthX = new double[nvx];
	LengthY = new double[nvy];
	LengthZ = new double[nvz];
	
	double* intX;
	double* intY;
	double* intZ;
	
	intX = new double[2];
	intY = new double[2];
	intZ = new double[2];
	
	for (int i = 0 ; i < nvx ; i++){
	LengthX[i] = i * (1.);
	}
	for (int i = 0 ; i < nvy ; i++){
	LengthY[i] = i * (1.);
	}
	for (int i = 0 ; i < nvz ; i++){
	LengthZ[i] = i * (1.);
	}	
	
	double lbz,lby,lbx;
	for (int z = 0 ; z < nvz ; z++){
	lbz = LengthZ[z];
	for (int y = 0 ; y < nvy ; y++){
	lby = LengthY[y];
	for (int x = 0 ; x < nvx ; x++){
	lbx = LengthX[x];
	
		for(int i = 0 ; i < nsiteX ; i++){
		intX[i] = lbx + i;
		}
		for(int i = 0 ; i < nsiteY ; i++){
		intY[i] = lby + i;
		}
		for(int i = 0 ; i < nsiteZ ; i++){
		intZ[i] = lbz + i;
		}
	
	double cc[3] = { compute_mean(intX,nsiteX),compute_mean(intY,nsiteY),compute_mean(intZ,nsiteZ) }; // wt 
	double sxx,sxy,sxz,syx,syy,syz,szx,szy,szz;
	sxx = sxy = sxz = syx = syy = syz = szx = szy = szz = 0.;
	
	for (int zz = 0 ; zz < nsiteZ ; zz++){
	for (int xx = 0 ; xx < nsiteX ; xx++){
	for (int yy = 0 ; yy < nsiteY ; yy++){
	
	double ci[3] = {intX[xx],intY[yy],intZ[zz]};
	double xci[3] = {cc[0]-ci[0],cc[1]-ci[1],cc[2]-ci[2]};
	xci[0] = xci[0] / sqrt(xci[0]*xci[0]+xci[1]*xci[1]+xci[2]*xci[2]) ;
	xci[1] = xci[1] / sqrt(xci[0]*xci[0]+xci[1]*xci[1]+xci[2]*xci[2]) ;
	xci[2] = xci[2] / sqrt(xci[0]*xci[0]+xci[1]*xci[1]+xci[2]*xci[2]) ;
	
	idn = ipx(x,y,z,nvx,nvy);
	idx = ipx(intX[xx],intY[yy],intZ[zz],LX,LY);
	sxx += xci[0] * fx[idx];
	sxy += xci[0] * fy[idx] + xci[1] * fx[idx];
	sxz += xci[0] * fz[idx] + xci[2] * fx[idx];
	syx += xci[1] * fx[idx] + xci[0] * fy[idx];
	syy += xci[1] * fy[idx];
	syz += xci[1] * fz[idx] + xci[2] * fy[idx];
	szx += xci[2] * fx[idx] + xci[0] * fz[idx];
	szy += xci[2] * fy[idx] + xci[1] * fz[idx];
	szz += xci[2] * fz[idx];
	}
	}
	}
	sxx = sxx/(nsiteX*nsiteY*nsiteZ);
	sxy = sxy/(2.*nsiteX*nsiteY*nsiteZ);
	sxz = sxz/(2.*nsiteX*nsiteY*nsiteZ);
	syx = syx/(2.*nsiteX*nsiteY*nsiteZ);
	syy = syy/(nsiteX*nsiteY*nsiteZ);
	syz = syz/(2.*nsiteX*nsiteY*nsiteZ);
	szx = szx/(2.*nsiteX*nsiteY*nsiteZ);
	szy = szy/(2.*nsiteX*nsiteY*nsiteZ);
	szz = szz/(nsiteX*nsiteY*nsiteZ);

	sigxx[idn] = sxx ;
	sigxy[idn] = sxy ;
	sigxz[idn] = sxz ;
	sigyy[idn] = syy ;
	sigyz[idn] = syz ;
	sigzz[idn] = szz ;
	
	}
	}
	}

	delete [] LengthX;
	delete [] LengthY; 
	delete [] LengthZ;
	
	delete [] intX;
	delete [] intY;
	delete [] intZ;

}
	

void resize_field(double data[], double datan[], int LX, int LY, int LZ, int LXn, int LYn, int LZn){
    
     int xn;
     int yn;
     int zn;
     int count = 0;
	for(int z = 0 ; z < LZn ; z++){
	for(int y = 0 ; y < LYn ; y++){
	for(int x = 0 ; x < LXn ; x++){
	
		if(z<LZ){
		zn = z;
		}
		if(x<LX){
		xn = x;
		}
		if(y<LY){
		yn = y;
		}
		if(z >= LZ){
		zn = z - LZ;
		}
		if(y >= LY){
		yn = y - LY;
		}
		if(x >= LX){
		xn = x - LX;
		}
		
	int idxn = ipx(x,y,z,LXn,LYn);
	int idx = ipx(xn,yn,zn,LX,LY);
	datan[idxn] = data[idx];
	count = count + 1;
	}
	}
	}
}

double GetXPosition(unsigned k, int LX, int LY) 
  { return (k/LY)%LX;}

double GetYPosition(unsigned k, int LY) 
  {return k%LY;}
  
double GetZPosition(unsigned k,int LX, int LY)  
  {return k/(LX*LY);}


void  import_fields( const string& _name, double data[], int LX, int LY,int LZ ){

	double fi;
	int tot = (LX*LY*LZ);
	fstream file(_name);
	for (int i = 0 ; i < tot ; i++){
	file >> fi;	
	data[i] = fi;
	}
	
}

void  export_fields(const string& _name, int num, double tmp[], size_t tot){
	
	ostringstream str1;
	str1 << num;
	string app1 = str1.str();
	string result = _name + app1;

   const char * c = result.c_str();
   FILE * sortie;
   sortie = fopen(c, "w+");	
   for (size_t k = 0 ; k < (tot) ; k++) fprintf(sortie,"%g\n",tmp[k]);
   fclose(sortie);
}

void CG_SIG_ARB(int LX, int LY, int LZ,
                               const double fx[], const double fy[], const double fz[],
                               double sigxx[], double sigxy[], double sigxz[],
                               double sigyy[], double sigyz[], double sigzz[],
                               int n_cg) {
    // Define the number of coarse cells along each axis.
    // Each coarse cell is obtained by integrating over an n_cg^3 block.
    // We assume that the coarse cell index x runs from 0 to (LX - n_cg),
    // so that there are nvx = LX - n_cg + 1 coarse cells in x.
    int nvx = LX - n_cg + 1;
    int nvy = LY - n_cg + 1;
    int nvz = LZ - n_cg + 1;
    
    // Loop over each coarse cell.
    for (int z = 0; z < nvz; z++) {
        for (int y = 0; y < nvy; y++) {
            for (int x = 0; x < nvx; x++) {
                // Compute the cell-center coordinates for this integration region.
                // For points x, x+1, ..., x+n_cg-1 the mean is x + (n_cg-1)/2.
                double cc_x = x + (n_cg - 1) / 2.0;
                double cc_y = y + (n_cg - 1) / 2.0;
                double cc_z = z + (n_cg - 1) / 2.0;
                
                // Initialize accumulators for the stress components.
                double sxx = 0, sxy = 0, sxz = 0, syy = 0, syz = 0, szz = 0;
                
                // Loop over the integration region of size n_cg x n_cg x n_cg.
                for (int dz = 0; dz < n_cg; dz++) {
                    for (int dy = 0; dy < n_cg; dy++) {
                        for (int dx = 0; dx < n_cg; dx++) {
                            double ix = x + dx;
                            double iy = y + dy;
                            double iz = z + dz;
                            
                            // Compute the fine-grid index (assuming fx,fy,fz are defined on an LX×LY×LZ grid)
                            int idx = ipx(ix, iy, iz, LX, LY);
                            
                            // Compute the vector from the integration point to the coarse cell center.
                            double diff_x = cc_x - ix;
                            double diff_y = cc_y - iy;
                            double diff_z = cc_z - iz;
                            double norm = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
                            if (norm == 0) norm = 1; // safeguard against division by zero
                            double ux = diff_x / norm;
                            double uy = diff_y / norm;
                            double uz = diff_z / norm;
                            
                            // Accumulate stress contributions.
                            sxx += ux * fx[idx];
                            // For off-diagonal terms, note the original code averages with a factor 1/2.
                            sxy += ux * fy[idx] + uy * fx[idx];
                            sxz += ux * fz[idx] + uz * fx[idx];
                            syy += uy * fy[idx];
                            syz += uy * fz[idx] + uz * fy[idx];
                            szz += uz * fz[idx];
                        }
                    }
                }
                
                // Compute the normalization factor.
                double volume = static_cast<double>(n_cg * n_cg * n_cg);
                sxx /= volume;
                sxy /= (2.0 * volume);
                sxz /= (2.0 * volume);
                syy /= volume;
                syz /= (2.0 * volume);
                szz /= volume;
                
                // Determine the coarse cell's linear index.
                // The coarse grid has dimensions (nvx, nvy, nvz).
                int idn = ipx(x, y, z, nvx, nvy);
                
                // Store the coarse-grained stress components.
                sigxx[idn] = sxx;
                sigxy[idn] = sxy;
                sigxz[idn] = sxz;
                sigyy[idn] = syy;
                sigyz[idn] = syz;
                sigzz[idn] = szz;
            }
        }
    }
}

void CG_SIG(int LX, int LY, int LZ,
                               const double fx[], const double fy[], const double fz[],
                               double sigxx[], double sigxy[], double sigxz[],
                               double sigyy[], double sigyz[], double sigzz[]) {
    // nvx, nvy, nvz: number of coarse-grid cells along x, y, z (each cell spans 2 sites)
    int nvx = LX - 1;
    int nvy = LY - 1;
    int nvz = LZ - 1;
    const int nsite = 2;  // fixed integration points per direction

    // Loop over each coarse cell in the domain.
    for (int z = 0; z < nvz; z++) {
        for (int y = 0; y < nvy; y++) {
            for (int x = 0; x < nvx; x++) {
                // The "cell center" for the integration is at (x+0.5, y+0.5, z+0.5)
                double cx = x + 0.5;
                double cy = y + 0.5;
                double cz = z + 0.5;
                
                // Initialize stress accumulators.
                double sxx = 0, sxy = 0, sxz = 0, syy = 0, syz = 0, szz = 0;
                
                // Loop over the 2×2×2 integration points (unrolled for clarity)
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            // Compute the integration point coordinates:
                            double ix = x + dx;  // site coordinate along x
                            double iy = y + dy;  // site coordinate along y
                            double iz = z + dz;  // site coordinate along z
                            
                            // Compute the linear index for the force fields using a helper function.
                            // Assume ipx(a, b, c, NX, NY) maps (a,b,c) to a 1D index for an array of dimensions NX×NY×?
                            int idx = ipx(ix, iy, iz, LX, LY);
                            
                            // For each integration point, its "local coordinate" relative to the cell center:
                            double diff_x = cx - ix;
                            double diff_y = cy - iy;
                            double diff_z = cz - iz;
                            double norm = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
                            if (norm == 0) norm = 1;  // safeguard against division by zero
                            double ux = diff_x / norm;
                            double uy = diff_y / norm;
                            double uz = diff_z / norm;
                            
                            // Accumulate contributions.
                            // The following formulas mimic the original weighted sums:
                            sxx += ux * fx[idx];
                            sxy += ux * fy[idx] + uy * fx[idx];
                            sxz += ux * fz[idx] + uz * fx[idx];
                            syy += uy * fy[idx];
                            syz += uy * fz[idx] + uz * fy[idx];
                            szz += uz * fz[idx];
                        }
                    }
                }
                // Average over the 8 integration points.
                double factor = 8.0;
                // Note: some components were divided by 2 in the original code.
                sigxx[ipx(x, y, z, nvx, nvy)] = sxx / factor;
                sigxy[ipx(x, y, z, nvx, nvy)] = sxy / (2.0 * factor);
                sigxz[ipx(x, y, z, nvx, nvy)] = sxz / (2.0 * factor);
                sigyy[ipx(x, y, z, nvx, nvy)] = syy / factor;
                sigyz[ipx(x, y, z, nvx, nvy)] = syz / (2.0 * factor);
                sigzz[ipx(x, y, z, nvx, nvy)] = szz / factor;
            }
        }
    }
}


	
int main (){

	int LX = 64;
	int LY = 64;
	int LZ = 40;
	
	int cgLx = 1;
	int cgLy = 1;
	int cgLz = 1;
	
	double* fx;
	double* fy;
	double* fz;
	
	fx = new double[LX*LY*LZ];
	fy = new double[LX*LY*LZ];
	fz = new double[LX*LY*LZ];
	
	import_fields("fx_39.dat",fx,LX,LY,LZ);
	import_fields("fy_39.dat",fy,LX,LY,LZ);
	import_fields("fz_39.dat",fz,LX,LY,LZ);

	int LXn = LX - 1;
	int LYn = LY - 1;
	int LZn = LZ - 1;
			
	double* sigxx;
	double* sigxy;
	double* sigxz;
	double* sigyy;
	double* sigyz;
	double* sigzz;
	
	sigxx = new double[LXn*LYn*LZn];
	sigxy = new double[LXn*LYn*LZn];
	sigxz = new double[LXn*LYn*LZn];
	sigyy = new double[LXn*LYn*LZn];
	sigyz = new double[LXn*LYn*LZn];
	sigzz = new double[LXn*LYn*LZn];
	
	// coare_grain_stress_field( LX, LY, LZ, fx, fy, fz, sigxx, sigxy, sigxz, sigyy, sigyz, sigzz );
	CG_SIG( LX, LY, LZ, fx, fy, fz, sigxx, sigxy, sigxz, sigyy, sigyz, sigzz );
	// CG_SIG_ARB( LX, LY, LZ, fx, fy, fz, sigxx, sigxy, sigxz, sigyy, sigyz, sigzz,4);
	export_fields("sigxx_",39,sigxx,(LXn*LYn*LZn));
	export_fields("sigxy_",39,sigxy,(LXn*LYn*LZn));
	export_fields("sigxz_",39,sigxz,(LXn*LYn*LZn));
	export_fields("sigyy_",39,sigyy,(LXn*LYn*LZn));
	export_fields("sigyz_",39,sigyz,(LXn*LYn*LZn));
	export_fields("sigzz_",39,sigzz,(LXn*LYn*LZn));

	delete [] fx;
	delete [] fy;
	delete [] fz;

	delete [] sigxx;
	delete [] sigxy;
	delete [] sigxz;
	delete [] sigyy;
	delete [] sigyz;
	delete [] sigzz;
	
     return 0;
}







