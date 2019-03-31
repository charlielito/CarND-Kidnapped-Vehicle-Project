/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <random> // Need this for sampling from distributions

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

using namespace std;

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y);
int get_closest_landmark_index(LandmarkObs obs, std::vector<LandmarkObs> map_landmarks);

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  num_particles = 10;  // Set the number of particles

  std::default_random_engine gen; // is the random engine

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std[0]);
  
  // Create normal distributions for y and theta
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;
    
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle particle;
    particle.id = i;
    particle.x = sample_x;
    particle.y = sample_y;
    particle.theta = sample_theta;
    particle.weight = 1;

    particles.push_back(particle);   
     
    // Print your samples to the terminal.
    std::cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " 
              << sample_theta << std::endl;
  }

  std::vector<double> weights_(num_particles,1.0);
  weights = weights_;
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen; // is the random engine

  for(Particle &particle: particles){  
    double x_pred, y_pred, theta_pred;

    if (abs(yaw_rate) == 0){
      x_pred = particle.x + velocity*delta_t*cos(particle.theta);
      y_pred = particle.y + velocity*delta_t*sin(particle.theta);
      theta_pred = particle.theta;
    }
    else{
      double delta_yaw = yaw_rate*delta_t;
      x_pred = particle.x + velocity/yaw_rate*(sin(particle.theta + delta_yaw) - sin(particle.theta));
      y_pred = particle.y + velocity/yaw_rate*(cos(particle.theta) - cos(particle.theta + delta_yaw));
      theta_pred = particle.theta + delta_yaw;
    }

    //  normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x_pred, std_pos[0]);
    
    // Create normal distributions for y and theta
    normal_distribution<double> dist_y(y_pred, std_pos[1]);
    normal_distribution<double> dist_theta(theta_pred, std_pos[2]);

    // Update position with this random noise
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);

    // std::cout << "Prediction: " << particle.x << " " << particle.y << " " << particle.theta << std::endl;
  }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


  double weight_norm = 0;
  for(Particle &particle: particles){ 
    double weight = 1.0;

    std::vector<LandmarkObs> landmarks;
    // Filter landmarks that are out of range of particle
    for (auto &landmark: map_landmarks.landmark_list) {
        if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) > sensor_range) {
            continue;
        }
        LandmarkObs obs;
        obs.id = landmark.id_i;
        obs.x = landmark.x_f;
        obs.y = landmark.y_f;
        landmarks.push_back(obs);
    }

    for (LandmarkObs observation: observations){
      LandmarkObs observation_map;
      // transform to map x coordinate
      observation_map.x = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);

      // transform to map y coordinate
      observation_map.y = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);

      observation_map.id = observation.id;

      // Get index of closest landmark to this obs
      int index = get_closest_landmark_index(observation_map, landmarks);

      // Calculate this weight
      weight *= multiv_prob(std_landmark[0], std_landmark[1], 
                                    observation_map.x, observation_map.y,
                                    landmarks.at(index).x, 
                                    landmarks.at(index).y);

      // std::cout << "OBS: " << observation_map.x << "," << map_landmarks.landmark_list.at(index).x_f << \
      // " " << observation_map.y << "," << map_landmarks.landmark_list.at(index).y_f << \
      // " weight: " << weight << std::endl;

    }
    particle.weight = weight;
    weight_norm += weight;
  }

  // Normalize weights of particles
  int i=0;
  for (Particle &particle: particles){
    particle.weight /= weight_norm;
    // std::cout << "Particle: " << particle.x << " " << particle.y << " " << particle.weight << std::endl;
    weights[i] = particle.weight;
    i++;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::default_random_engine gen;
  std::discrete_distribution<> d_distribution(weights.begin(),weights.end());
  std::vector<Particle> gen_particles;
  
  for (int i = 0; i < num_particles; i++){
    int index = d_distribution(gen);
    gen_particles.push_back(particles[index]);
  }
  particles = gen_particles;

}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y)
  {
      // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
  }

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


int get_closest_landmark_index(LandmarkObs obs, std::vector<LandmarkObs> map_landmarks){
  double min_distance = INFINITY; // big number
  int index = 0;
  for(unsigned int i=0; i < map_landmarks.size(); i++){
    double distance = dist(obs.x, obs.y, map_landmarks.at(i).x, map_landmarks.at(i).y);
    if (distance < min_distance){
      index = i;
      min_distance = distance;
    }
  }
  return index;
}