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

using namespace std;

double Model::random_real(double min, double max)
{
  return uniform_real_distribution<>(min, max)(gen);
}

double Model::random_normal(double sigma)
{
  return normal_distribution<>(0., sigma)(gen);
}

double Model::random_normal_full(double mean, double sigma)
{
  return normal_distribution<>(mean, sigma)(gen);
}

unsigned Model::random_geometric(double p)
{
  return geometric_distribution<>(p)(gen);
}

unsigned Model::random_unsigned()
{
  return gen();
}

void Model::InitializeRandomNumbers()
{
  if(not set_seed)
  {
    // 'Truly random' device to generate seed
    std::random_device rd;
    seed = rd();
  }

  gen.seed(seed);
}

unsigned Model::random_poisson(double lambda)
{
  return poisson_distribution<>(lambda)(gen);
}

int Model::random_exponential(double lambda)
{
  return exponential_distribution<>(lambda)(gen);
}

double Model::random_uniform()
{

  return uniform_real_distribution<>(0.0, 2.0 * M_PI)(gen);

}

int Model::random_int_uniform(int min, int max){
return uniform_int_distribution<>(min, max)(gen);
}

double Model::random_double_uniform(double min, double max) {
    return std::uniform_real_distribution<double>(min, max)(gen);
}
