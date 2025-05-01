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
#include <random>  // for lognormal_distribution

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

// Generate a log-normal random value given desired mean and standard deviation in linear space
// mean_lin = desired E[X], sigma_lin = desired SD[X]
// Converts to underlying normal parameters mu and sigma_n:
//   sigma_n = sqrt(log(1 + (sigma_lin^2)/(mean_lin^2)))
//   mu     = log(mean_lin) - 0.5 * sigma_n^2
// Then uses lognormal_distribution(mu, sigma_n)
// Always positive support
double Model::random_lognormal(double mean_lin, double sigma_lin)
{
    double variance_lin = sigma_lin * sigma_lin;
    double mu = log(mean_lin) - 0.5 * log(1.0 + variance_lin / (mean_lin * mean_lin));
    double sigma_n = sqrt(log(1.0 + variance_lin / (mean_lin * mean_lin)));
    return lognormal_distribution<>(mu, sigma_n)(gen);
}

// Generate a log-normal random value with a minimum threshold min_lin
// Uses rejection sampling: resample until value >= min_lin
double Model::random_lognormal(double mu, double sigma_n, double min_lin)
{
    double value;
    // double variance_lin = sigma_lin * sigma_lin;
    // double mu = log(mean_lin) - 0.5 * log(1.0 + variance_lin / (mean_lin * mean_lin));
    // double sigma_n = sqrt(log(1.0 + variance_lin / (mean_lin * mean_lin)));
    lognormal_distribution<> dist(mu, sigma_n);
    do {
        value = dist(gen);
    } while (value < min_lin);
    return value;
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

int Model::random_int_uniform(int min, int max)
{
  return uniform_int_distribution<>(min, max)(gen);
}

double Model::random_double_uniform(double min, double max)
{
  return uniform_real_distribution<double>(min, max)(gen);
}
