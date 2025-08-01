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
#include "files.hpp"

using namespace std;

void Model::Write_phi(unsigned t){
  
  
  for(unsigned n=nstart; n<nphases; ++n)
  {
    const string oname = inline_str(output_dir, "cell_phi_", n, "_t_",t, ".dat");
    const char * cname = oname.c_str();    
    FILE * sortie;
    sortie = fopen(cname, "w+");	
    
    // update only patch (in parallel, each node to a different core)
    // PRAGMA_OMP(omp parallel for num_threads(nthreads) if(nthreads))
    for(unsigned q=0; q<patch_N; ++q){
    const coord tmp = GetNodePosOnDomain(n,q);          
    fprintf(sortie,"%.4u %u %u %u %.4e\n",q,tmp[0],tmp[1],tmp[2],phi[n][q]);      
    }
    fclose(sortie);    
  }
  // fclose(sortie);
}


void Model::Write_dphi(unsigned t){

  for(unsigned n=nstart; n<nphases; ++n)
  {
    const string oname = inline_str(output_dir, "cell_dphi_", n, "_t_",t, ".dat");
    const char * cname = oname.c_str();    
    FILE * sortie;
    sortie = fopen(cname, "w+");	
    
    // update only patch (in parallel, each node to a different core)
    // PRAGMA_OMP(omp parallel for num_threads(nthreads) if(nthreads))
    for(unsigned q=0; q<patch_N; ++q){
    const coord tmp = GetNodePosOnDomain(n,q);          
      fprintf(sortie,"%.4u %u %u %u %.4e %.4e\n",q,tmp[0],tmp[1],tmp[2],dphi[n][q],dphi[n][q]);      
    }
    fclose(sortie);    
  }
  // fclose(sortie);
}



void Model::Write_COM(unsigned t){

    const string fname = "center_of_mass.dat";
    const char * cname = fname.c_str();
    FILE * sortie;
    sortie = fopen(cname, "a");    
  
  for(unsigned n=0; n<nphases_index.size(); ++n)
  {
    fprintf( sortie,"%u %.4e %.4e %.4e \n",t,com[n][0],com[n][1],com[n][2]);      
  }
    fclose(sortie);    
  
}

void Model::Write_divAngle(unsigned t, unsigned n, unsigned i, bool mutate, double angle, double plocal, double pcomp, double ptens) {
    const std::string fname = "division_angles.dat";
    const char *cname = fname.c_str();
    FILE *sortie = fopen(cname, "a");
    if (sortie != nullptr) {
        // Using %u for unsigned integers, %d for the bool (cast to int), and %g for the double.
        fprintf(sortie, "%u %u %u %u %d %g %g %g %g\n", t, n, i, nphases, static_cast<int>(mutate), angle,plocal,pcomp,ptens);
        fclose(sortie);
    } else {
        // Handle error: could not open file.
        fprintf(stderr, "Error opening file %s\n", cname);
    }
}


void Model::Write_OU(unsigned t,unsigned i) {

  //for(unsigned n=0; n<nphases_index.size(); ++n){
  unsigned n = nphases_index[i];
  const string oname = inline_str("cell_", n,".dat");
  const char * cname = oname.c_str();    
  FILE * sortie;
  sortie = fopen(cname, "a");
  fprintf(sortie, "%u %u %u %g %g %g\n",i,n,t,timer[i],divisiontthresh[i],stored_tmean[i]);
  fclose(sortie);
  //}
}




void Model::Write_contArea(unsigned t){
        
    const string fname = "contact_area.dat";
    const char * cname = fname.c_str();
    FILE * sortie;
    sortie = fopen(cname, "a");   
    
    for(unsigned n=0; n<nphases_index.size(); ++n){
    double aL0 = 0.;
    double aL1 = 0.;
    double aL2 = 0.;
    // PRAGMA_OMP(omp parallel for num_threads(nthreads) if(nthreads))
    for(unsigned q=0; q<patch_N; ++q){ 
    const coord GlobCoor = GetNodePosOnDomain(n,q);   
	if ( GlobCoor[2] == wall_thickness ) aL0 += phi[n][q] * phi[n][q];
	if ( GlobCoor[2] == wall_thickness + 1 ) aL1 += phi[n][q] * phi[n][q];
	if ( GlobCoor[2] == wall_thickness + 2 ) aL2 += phi[n][q] * phi[n][q]; 
    }
    fprintf(sortie, "%u %.4e %.4e %.4e %.4e\n", t, aL0,aL1,aL2, vol[n]);
    }
    fclose(sortie);
}

void Model::Write_Density(unsigned t){
        
    const string fname = "density.dat";
    const char * cname = fname.c_str();
    FILE * sortie;
    sortie = fopen(cname, "a");   
    
    double aL0 = 0.;
    double aL1 = 0.;
    double aL2 = 0.;
    for(unsigned n=0; n<nphases_index.size(); ++n){

    // PRAGMA_OMP(omp parallel for num_threads(nthreads) if(nthreads))
    for(unsigned q=0; q<patch_N; ++q){ 
    const coord GlobCoor = GetNodePosOnDomain(n,q);   
	if ( GlobCoor[2] == wall_thickness ) aL0 += phi[n][q] * phi[n][q];
	if ( GlobCoor[2] == wall_thickness + 1 ) aL1 += phi[n][q] * phi[n][q];
	if ( GlobCoor[2] == wall_thickness + 2 ) aL2 += phi[n][q] * phi[n][q];
    }
    }
    fprintf(sortie, "%u %.4e %.4e %.4e \n", t, aL0/(Size[0]*Size[1]),aL1/(Size[0]*Size[1]),aL2/(Size[0]*Size[1]) );
    
    fclose(sortie);
}


void Model::Write_visData(unsigned t){
    
    field pfVis;
    pfVis.resize(N,0.);    
    
    for(unsigned n=nstart; n<nphases; ++n)
    {
        //PRAGMA_OMP(omp parallel for num_threads(nthreads) if(nthreads))
        for(unsigned q=0; q<patch_N; ++q){ 
        const coord GlobCoor = GetNodePosOnDomain(n,q);   
	 // pfVis[GlobCoor[1] + Size[1]*GlobCoor[0] + Size[0]*Size[1]*GlobCoor[2]]  += phi[n][q];
	 pfVis[GlobCoor[0] + Size[0]*GlobCoor[1] + Size[0]*Size[1]*GlobCoor[2]]  += phi[n][q];
        }
        
    }
    
    
    FILE * sortie;
    char nomfic[256];
    sprintf(nomfic, "tissue_%01u.vtk", t);
    
    sortie = fopen(nomfic, "w");
    fprintf(sortie, "# vtk DataFile Version 2.0\n");
    fprintf(sortie, "volume example\n");
    fprintf(sortie, "ASCII\n");
    fprintf(sortie, "DATASET STRUCTURED_POINTS\n");    
    fprintf(sortie, "DIMENSIONS %i %i %i\n", Size[0], Size[1], Size[2]);
    fprintf(sortie, "ASPECT_RATIO  %i %i %i\n", 1, 1, 1);
    fprintf(sortie, "ORIGIN  %i %i %i\n", 0, 0, 0);    
    fprintf(sortie, "POINT_DATA  %i\n", N);
    fprintf(sortie, "SCALARS volume_scalars double 1\n");       
    fprintf(sortie, "LOOKUP_TABLE default \n");       
    size_t ik = 0;
    for (size_t z = 0 ; z < Size[2] ; z++) {
        for (size_t y = 0 ; y < Size[1] ; y++) {
            for (size_t x = 0 ; x < Size[0] ; x++) {
                fprintf(sortie, " %g ", pfVis[ik++]);
            }
        }
        fprintf(sortie,"\n");
    }
    
    fprintf(sortie, "SCALARS id float\n");
    fprintf(sortie, "LOOKUP_TABLE default \n");       
    ik = 0;
    for (size_t z = 0 ; z < Size[2] ; z++) {
        for (size_t y = 0 ; y < Size[1] ; y++) {
            for (size_t x = 0 ; x < Size[0] ; x++) {
                fprintf(sortie, " %g ", pfVis[ik++]);
            }
        }
        fprintf(sortie,"\n");
    }
    
    fclose(sortie);
     
}


void Model::Write_velocities(unsigned t){

    const string fname = "velocities_out.dat";
    const char * cname = fname.c_str();
    FILE * sortie;
    sortie = fopen(cname, "a");    
  
  for(unsigned n=0; n<nphases_index.size(); ++n)
  {
        
    fprintf(sortie,"%u %.4e %.4e %.4e\n",t,velocity[n][0],velocity[n][1],velocity[n][2]);      
  }
    fclose(sortie);    
  
}

void Model::Write_forces(unsigned t){
    
    const string fname = "forces_out.dat";
    const char * cname = fname.c_str();
    FILE * sortie;
    sortie = fopen(cname, "a");    
  
  for(unsigned n=nstart; n<nphases_index.size(); ++n)
  {
        
    fprintf(sortie,"%u %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",t,Fpressure[n][0],Fpressure[n][1],Fpressure[n][2],Fnem[n][0],Fnem[n][1],Fnem[n][2],Fshape[n][0],Fshape[n][1],Fshape[n][2],Fpol[n][0],Fpol[n][1],Fpol[n][2]   );      
  }
    fclose(sortie);    
  
}


void Model::WriteFrame(unsigned t)
{
  // construct output name
  const string oname = inline_str(output_dir, "frame", t, ".json");

  // write
  {
    stringstream buffer;
    {
      oarchive ar(buffer, "frame", 1);
      // serialize
      SerializeFrame(ar);

     if(ar.bad_value()) throw error_msg("nan found while writing file.");
    }
    // dump to file
    std::ofstream ofs(oname.c_str(), ios::out);
    ofs << buffer.rdbuf();
  }

  // compress
  if(compress) compress_file(oname, oname);
  if(compress_full) compress_file(oname, runname);
}

void Model::WriteParams()
{
  // a name that makes sense
  const string oname = inline_str(output_dir, "parameters.json");

  // write
  {
    stringstream buffer;
    {
      // serialize
      oarchive ar(buffer, "parameters", 1);
      // ...program parameters...
      ar & auto_name(Size)
         & auto_name(BC)
         & auto_name(nsteps)
         & auto_name(nsubsteps)
         & auto_name(ninfo)
         & auto_name(nstart);
      // ...and model parameters
      SerializeParameters(ar);

     if(ar.bad_value()) throw error_msg("nan found while writing file.");
    }
    // dump to file
    std::ofstream ofs(oname.c_str(), ios::out);
    ofs << buffer.rdbuf();
  }

  // compress
  if(compress) compress_file(oname, oname);
  if(compress_full) compress_file(oname, runname);
}

void Model::ClearOutput()
{
  if(compress_full)
  {
    // file name of the output file
    const string fname = runname + ".zip";

    {
      // try open it
      ifstream infile(fname);
      // does not exist we are fine
      if(not infile.good()) return;
    }

    if(not force_delete)
    {
      // ask
      char answ = 0;
      cout << " remove output file '" << fname << "'? ";
      cin >> answ;

      if(answ != 'y' and answ != 'Y')
        throw error_msg("output file '", fname,
                        "' already exist, please provide a different name.");
    }

    // delete
    remove_file(fname);
  }
  else
  {
    // extension of single files
    string ext = compress ? ".json.zip" : ".json";

    // check that parameters.json does not exist in the output dir and if it
    // does ask for deletion (this is not completely fool proof, but ok...)
    {
      ifstream infile(output_dir + "parameters" + ext);
      if(not infile.good()) return;
    }

    if(not force_delete)
    {
      // ask
      char answ = 0;
      cout << " remove output files in directory '" << output_dir << "'? ";
      cin >> answ;

      if(answ != 'y' and answ != 'Y')
        throw error_msg("output files already exist in directory '",
                        output_dir, "'.");
    }

    // delete all output files
    remove_file(output_dir);
  }
}

void Model::CreateOutputDir()
{
  // if full compression is on: we need to create a random tmp directory
  if(compress_full)
  {
    // use hash of runname string plus salt
    hash<string> hash_fn;
    unsigned dir_name = hash_fn(inline_str(runname, random_unsigned()));
    output_dir = inline_str("/tmp/", dir_name, "/");
  }
  // if full compression is off: just dump files where they belong
  else
    // note that runname can not be empty from options.cpp
    output_dir = runname + ( runname.back()=='/' ? "" : "/" );

  // clear output if needed
  ClearOutput();

  // create output dir if needed
  create_directory(output_dir);
}
