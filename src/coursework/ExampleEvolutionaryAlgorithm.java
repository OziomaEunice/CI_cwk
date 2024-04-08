package coursework;

import java.util.ArrayList;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet; 
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {
	

	// Define Leaky ReLU with samll slope for negative values
	private static final double Alpha = 0.1;
	
	
	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);
		
		// see completed generations
		int completedGenerations = 0;

		/**
		 * main EA processing loop
		 */		
		
		while (evaluations < Parameters.maxEvaluations) {

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */

			// Select 2 Individuals from the current population.
			Individual parent1 = tournamentSelect(); 
			Individual parent2 = tournamentSelect();

			// Generate a child by crossover			
			ArrayList<Individual> children = onePCrossover(parent1, parent2);
			//ArrayList<Individual> children = twoPCrossover(parent1, parent2);
			//ArrayList<Individual> children = uniformCrossover(parent1, parent2);			
			
			
			//mutate the offspring
			mutate(children);
			
			// Evaluate the children
			evaluateIndividuals(children);			

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations	
			completedGenerations++;
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * SELECTION -- using the Tournament selection
	 */
	private Individual tournamentSelect() {		
		// select the individuals (which are participants) from the population
		// and perform a tournament amongst them
		ArrayList<Individual> participants = new ArrayList<>();
		for(int i = 0; i < Parameters.popSize; i++) {
			participants.add(population.get(Parameters.random.nextInt(Parameters.popSize)));
		}
		
		
		// find the fittest individual
		// and if the individual (or participant) has the fittest 
		// chromosome, select it
		Individual fittest = null;
		for(Individual participant : participants) {
			if(fittest == null || participant.fitness < fittest.fitness) {
				fittest = participant;
			}
		}
		
		return fittest.copy(); // return a copy of the fittest individual
	}

	/**
	 * CROSSOVER / REPRODUCTION  -- Using One-Point Crossover
	 * 
	 * **/
	private ArrayList<Individual>  onePCrossover(Individual parent1, Individual parent2){
		
		ArrayList<Individual> children = new ArrayList<>();
		
		int crossPoint = Parameters.random.nextInt(parent1.chromosome.length); // choose crosspoint randomly
		
		Individual child = new Individual();
					
		// copy first part of child from parent1, and second part
		// from parent2
		for (int j = 0; j < crossPoint; j++) {
			child.chromosome[j] = parent1.chromosome[j];
		}
		for (int j = crossPoint; j < parent1.chromosome.length; j++) {
			child.chromosome[j] = parent2.chromosome[j];
		}
		
		children.add(child);
		
		return children;
	}
	
	
	/**
	 * -- Using Two-Point Crossover
	 */
	private ArrayList<Individual> twoPCrossover(Individual parent1, Individual parent2){
		
		ArrayList<Individual> children = new ArrayList<>();

		// choose 1st and 2nd crosspoints randomly
		int crossPoint1 = Parameters.random.nextInt(parent1.chromosome.length); 
		int crossPoint2 = Parameters.random.nextInt(parent1.chromosome.length); 
		
		Individual child = new Individual();
		
		// to ensure that the crossover point between the
		// chromosomes are different
		while (crossPoint1 == crossPoint2) {crossPoint2 = Parameters.random.nextInt(parent1.chromosome.length);}
		
		// ensure that the first crosspoint comes before the second one
		if (crossPoint1 > crossPoint2) {
			int temporal = crossPoint1;
			crossPoint1 = crossPoint2;
			crossPoint2 = temporal;
		}
		
		// create two child individuals using the generated chromosomes 
		// and add them to the list of children
		Individual children1 = new Individual();
		Individual children2 = new Individual();
		
		for(int i = 0; i < parent1.chromosome.length; i++) {
			if(i < crossPoint1 || i >= crossPoint2) {
				children1.chromosome[i] = parent1.chromosome[i];
				children2.chromosome[i] = parent2.chromosome[i];
			} else {
				children1.chromosome[i] = parent2.chromosome[i];
				children2.chromosome[i] = parent1.chromosome[i];
			}
		}
		
		
		
		children.add(children1);
		children.add(children2);	
		
		return children;
	}
	
	
	/**
	 * -- using the Uniform Crossover
	 */
	private ArrayList<Individual> uniformCrossover(Individual parent1, Individual parent2) {
		// create a list to store the offspring (children)
		ArrayList<Individual> children = new ArrayList<>();
		
		// uniform crossover: select each gene (bit) from one of the corresponding
		// genes of the parents' chromosomes
		// create two arrays to store the chromosome of children
		double[] child1 = new double[parent1.chromosome.length];
		double[] child2 = new double[parent2.chromosome.length];
		
		
		// iterate each gene (or element) in individual parents' chromosome.
		// for each gene, select randomly one of the parent's gene to be inherited 
		// by each child
		for(int i = 0; i < parent1.chromosome.length; i++) {
			if(Parameters.random.nextBoolean()) {
				child1[i] = parent1.chromosome[i];
				child2[i] = parent2.chromosome[i];
			} else {
				child1[i] = parent2.chromosome[i];
				child2[i] = parent1.chromosome[i];
			}
		}
		
		// create two child individuals using the generated chromosomes 
		// and add them to the list of children
		Individual children1 = new Individual();
		Individual children2 = new Individual();
		
		children.add(children1);
		children.add(children2);	
		
		return children;
	}
	
	
	
	
	
	/**
	 * Mutation -- using non-Uniform Mutation
	 * 
	 * 
	 */
//	private void nonUniformMutation(ArrayList<Individual> individuals) {
//		// for each individual's gene in the chromosome, if a randomly
//		// generated value is less that the mutation rate, change the mutation
//		// process.
//		for(Individual individual : individuals) {
//			for (int i = 0; i < individual.chromosome.length; i++) {
//				if(Parameters.random.nextDouble() < Parameters.mutateRate) {
//					
//					// change mutation and add the mutation change
//					double changeMutation = Parameters.random.nextGaussian() * Parameters.scale * Parameters.mutateChange;
//					individual.chromosome[i] += changeMutation;
//					
//					
//					// to ensure that the gene value remains within the boundaries
//					// set out for it
//					individual.chromosome[i] = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, individual.chromosome[i]));
//				}
//			}
//		}
//	}
	
	/**
	 * Mutation -- using a Uniform Mutation
	 */
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(ArrayList<Individual> individuals) {		
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(ArrayList<Individual> individuals) {
		 // calculate and store the index of the worst individual
		// in the current population
		int worstIndex = getWorstIndex();
		
		// for each individual in the population, if it has a fitness value
		// less than the fitness value of the worst individual in the population
		// then replace it with the fitness of the worst individual, 
		// else leave the current individual
		for(Individual individual : individuals) {
			if(individual.fitness < population.get(worstIndex).fitness) {
				// individual.fitness = population.get(worstIndex).fitness;
				population.set(worstIndex, individual);
			}
		}		
	}
	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

//	@Override
//	public double activationFunction(double x) {
//		if (x < -20.0) {
//			return -1.0;
//		} else if (x > 20.0) {
//			return 1.0;
//		}
//		return Math.tanh(x);
//	}
	
	
	// using Leaky ReLU as the activation function	
	@Override
	public double activationFunction(double x) {
		return x >= 0 ? x : Alpha * x;
	}
}
