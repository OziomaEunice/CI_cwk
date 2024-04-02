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

			// Select 2 Individuals from the current population. Currently returns random Individual
			Individual parent1 = tournamentSelect(); 
			Individual parent2 = tournamentSelect();

			// Generate a child by crossover. Not Implemented			
			ArrayList<Individual> children = uniformCrossover(parent1, parent2);			
			
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
		// select the parents individuals (which are participants) from the population
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
	 * CROSSOVER / REPRODUCTION  -- using the Uniform Crossover
	 */
	private ArrayList<Individual> uniformCrossover(Individual parent1, Individual parent2) {
		// create a list to store the offspring (children)
		ArrayList<Individual> children = new ArrayList<>();
		
		// uniform crossover: select each gene (bit) from one of the corresponding
		// genes of the parents' chromosomes
		// create two arrays to store the chromosome of children
		double[] child1 = new double[parent1.chromosome.length];
		double[] child2 = new double[parent2.chromosome.length];
		
		
		// iterate of each gene (or element) in individual parents' chromosome
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
		for(Individual individual : individuals) {
			int idx = getWorstIndex();		
			population.set(idx, individual);
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

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}
