//----------------------------------------------------------------------------------------------
//	Filename:	nbody.cpp
//	Author:		Keith Bugeja
//----------------------------------------------------------------------------------------------
//  CPS3236 assignment for academic year 2017/2018:
//	Sample naive [O(n^2)] implementation for the n-Body problem.
//----------------------------------------------------------------------------------------------

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

#include "vector2.h"

/*
 * Constant definitions for field dimensions, and particle masses
 */
const int fieldWidth = 1000;
const int fieldHalfWidth = fieldWidth >> 1;
const int fieldHeight = 1000;
const int fieldHalfHeight = fieldHeight >> 1;

const float minBodyMass = 2.5f;
const float maxBodyMassVariance = 5.f;

/*
 * Particle structure
 */
struct Particle
{
	Vector2 Position;
	Vector2 Velocity;
	float Mass;
	
	Particle(void) 
		: Position( ((float)rand()) / RAND_MAX * fieldWidth - fieldHalfWidth,
					((float)rand()) / RAND_MAX * fieldHeight - fieldHalfHeight)
		, Velocity( 0.f, 0.f )
		, Mass ( ((float)rand()) / RAND_MAX * maxBodyMassVariance + minBodyMass )
	{ }

	Particle(float m, float x, float y) 
		: Position( x, y )
		, Velocity( 0.f, 0.f )
		, Mass( m )
	{ }
};

/*
 * Compute forces of particles exerted on one another
 */
void ComputeForcesParallel(std::vector<Particle> &p_bodies, float p_gravitationalTerm, float p_deltaT)
{
	#pragma omp parallel
	{
		//thread private
		Vector2 force, acceleration;
		
		#pragma omp for schedule(dynamic)
		for (size_t j = 0; j < p_bodies.size(); ++j)
		{
			Particle &p1 = p_bodies[j];
		
			force = 0.f, acceleration = 0.f;

			// #pragma omp declare reduction (addForce:Vector2:omp_out += omp_in) initializer (omp_priv=omp_orig)
			// #pragma omp parallel for private(direction,distance) reduction(addForce:force)
			// for (size_t k = 0; k < p_bodies.size(); ++k)
			// {
			// 	if (k == j) continue;
			
			// 	Particle &p2 = p_bodies[k];
				
			// 	// Compute direction vector
			// 	direction = p2.Position - p1.Position;
				
			// 	// Limit distance term to avoid singularities
			// 	distance = std::max<float>( 0.5f * (p2.Mass + p1.Mass), fabs(direction.Length()) );
				
			// 	// Accumulate force
			// 	force += direction / (distance * distance * distance) * p2.Mass; 
			// }
			
			for (size_t k = 0; k < p_bodies.size(); ++k)
			{
				if (k == j) continue;
			
				Particle &p2 = p_bodies[k];
				
				// Compute direction vector
				Vector2 direction = p2.Position - p1.Position;
				
				// Limit distance term to avoid singularities
				float distance = std::max<float>( 0.5f * (p2.Mass + p1.Mass), fabs(direction.Length()) );
				
				// Accumulate force
				force += direction / (distance * distance * distance) * p2.Mass; 
			}

			// Compute acceleration for body 
			acceleration = force * p_gravitationalTerm;

			// Integrate velocity (m/s)
			p1.Velocity += acceleration * p_deltaT;
		}
	}
}

/*
 * Update particle positions
 */
void MoveBodiesParallel(std::vector<Particle> &p_bodies, float p_deltaT)
{	
	
	#pragma omp parallel for
	for (size_t j = 0; j < p_bodies.size(); ++j)
	{
		p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
	}
}

/*
 * Commit particle masses and positions to file in CSV format
 */
void PersistPositions(const std::string &p_strFilename, std::vector<Particle> &p_bodies)
{
	std::cout << "Writing to file: " << p_strFilename << std::endl;
	
	std::ofstream output(p_strFilename.c_str());
	
	if (output.is_open())
	{	
		for (int j = 0; j < p_bodies.size(); j++)
		{
			output << 	p_bodies[j].Mass << ", " <<
				p_bodies[j].Position.Element[0] << ", " <<
				p_bodies[j].Position.Element[1] << std::endl;
		}
		
		output.close();
	}
	else
		std::cerr << "Unable to persist data to file:" << p_strFilename << std::endl;

}

int main(int argc, char **argv)
{	
	omp_set_num_threads(omp_get_max_threads());

	// Set default values
	bool output = true;
	char *inputFile;
	int particleCount = 1024;
	int maxIteration = 1000;
	float deltaT = 0.005f;
	float gTerm = 1.f;

	// Parse command line arguments
	for(int i=1; i<argc; ++i)
    {
		if(strcmp("-f",argv[i]) == 0){
			inputFile = argv[i+1];
		}
		else if(strcmp("-o",argv[i]) == 0){
			if(strcmp("false",argv[i+1]) == 0){
				output = false;
			}
		}
		else if(strcmp("-b",argv[i]) == 0){
			int len;
			if(sscanf(argv[i+1], "%d %n", &particleCount, &len) != 1)
			{
				std::cerr << "Invalid argument: " << argv[i+1] << ". Using default value 1024 instead" << std::endl;
			}
		}
		else if(strcmp("-g",argv[i]) == 0){
			int len;
			if(sscanf(argv[i+1], "%f %n", &gTerm, &len) != 1)
			{
				std::cerr << "Invalid argument: " << argv[i+1] << ". Using default value 1.0 instead" << std::endl;
			}
		}
		else if(strcmp("-i",argv[i]) == 0){
			int len;
			if(sscanf(argv[i+1], "%d %n", &maxIteration, &len) != 1)
			{
				std::cerr << "Invalid argument: " << argv[i+1] << ". Using default value 1000 instead" << std::endl;
			}
		}
		else if(strcmp("-d",argv[i]) == 0){
			int len;
			if(sscanf(argv[i+1], "%f %n", &deltaT, &len) != 1)
			{
				std::cerr << "Invalid argument: " << argv[i+1] << ". Using default value 0.005 instead" << std::endl;
			}
		}
    }
	
	std::stringstream fileOutput;
	std::vector<Particle> bodies;

	std::ifstream infile(inputFile);
	// Check if file exists
	if(infile.is_open())
	{	
		std::cout << "Input File:" << inputFile << std::endl;
		if(!output) std::cout << "Output Suppressed" << std::endl;
		std::cout << "Gravity: " << gTerm << std::endl;
		std::cout << "Iterations: " << maxIteration << std::endl;
		std::cout << "Simulation Time Step: " << deltaT << std::endl;

		// Get particle mass, x Position, y Position from file
		std::string line;
		int i = 0;

		while (std::getline(infile, line))
		{	
			std::istringstream ss(line);
			std::string token;
			float mass;
			float xPos;
			float yPos;
			int j = 0;
			while (std::getline(ss, token, ','))
			{
			    switch(j)
				{
					case(0):
						mass = stof(token);
						break;
					case(1):
						xPos = stof(token);
						break;
					case(2):
						yPos = stof(token);
						break;
				}
				j++;
			}
			bodies.push_back(Particle(mass,xPos,yPos));
			i++;
		}
		std::cout << "Particles: " << i << std::endl;
	} 
	//If file does not exist set random values
	else
	{
		std::cout << "No input file. Initialising random values instead." << std::endl;
		if(!output) std::cout << "Output Suppressed" << std::endl;
		std::cout << "Gravity: " << gTerm << std::endl;
		std::cout << "Iterations: " << maxIteration << std::endl;
		std::cout << "Simulation Time Step: " << deltaT << std::endl;
		std::cout << "Particles: " << particleCount << std::endl;
		
		// Initialise particles with random values
		for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
		{
			bodies.push_back(Particle());
		}
	}

	for (int iteration = 0; iteration < maxIteration; ++iteration)
	{	
		// Start timing
		struct timeval start, end;
		long mtime, seconds, useconds;
		gettimeofday(&start, NULL);

		ComputeForcesParallel(bodies, gTerm, deltaT);
		MoveBodiesParallel(bodies, deltaT);
		
		if(output == true)
		{
			fileOutput.str(std::string());
			fileOutput << "/home/martin/Documents/HPCAssignment/OutputOMP/nbody_" << iteration << ".txt";
			PersistPositions(fileOutput.str(), bodies);
		}

		// Stop timing
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds / 1000.0) + 0.5;
		printf("Full Iteration: %ld milliseconds\n", mtime);
	}
}