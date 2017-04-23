package com.chatbot.kelp.tutorial;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.StringJoiner;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.data.manipulator.TreePairRelTagger;
import it.uniroma2.sag.kelp.data.representation.structure.filter.LexicalStructureElementFilter;
import it.uniroma2.sag.kelp.data.representation.structure.similarity.ExactMatchingStructureElementSimilarity;
import it.uniroma2.sag.kelp.data.representation.tree.node.filter.ContentBasedTreeNodeFilter;
import it.uniroma2.sag.kelp.data.representation.tree.node.filter.TreeNodeFilter;
import it.uniroma2.sag.kelp.data.representation.tree.node.similarity.ContentBasedTreeNodeSimilarity;
import it.uniroma2.sag.kelp.data.representation.tree.node.similarity.TreeNodeSimilarity;
import it.uniroma2.sag.kelp.kernel.cache.DynamicIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.DynamicIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.pairs.UncrossedPairwiseProductKernel;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.RbfKernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.solver.LibSvmSolver;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;

public class ExpMartin {
	public static final String treeRepresentation = "tree";

	public static void main(String[] args) throws Exception {
		if(args.length < 5){
			System.out.println("EXPECTED 5 ARGUMENTS:");
			System.out.println("1) prediction file outputh");
            System.out.println("2) C");
            System.out.println("3) path to train set");
            System.out.println("4) path to dev set");
            System.out.println("5) path to test set");
            System.out.println("6))(optional) cache size - defaults to size(train_set+dev_set+test_set)");
			System.exit(0);
		}
		long start = System.currentTimeMillis();
		System.out.println("INPUT ARGUMENTS:");
		for(String arg : args){
			System.out.println(arg);
		}		

		String predictionFileOutput = args[0];
		float c = Float.parseFloat(args[1]);
		
		/*
		 * LOADING DATASETS
		 */

		/*
		 * LOADING DATASETS
		 */
        String trainsetFile = args[2];
        String devsetFile = args[3];
        String testsetFile = args[4];
		SimpleDataset trainset = new SimpleDataset();
		trainset.populate(trainsetFile);
		
		SimpleDataset devset = new SimpleDataset();
		devset.populate(devsetFile);
		SimpleDataset testset = new SimpleDataset();
		testset.populate(testsetFile);
		
		
		SimpleDataset completeDataset = new SimpleDataset();
		completeDataset.addExamples(trainset);
		completeDataset.addExamples(devset);
		completeDataset.addExamples(testset);
		
		
		//BEGIN REL TAGGING (see section 3.2 of the paper)
		HashSet<String> stopwords = new HashSet<String>();
		stopwords.add("be");
		stopwords.add("have");

		HashSet<String> posOfInterest = new HashSet<String>();
		posOfInterest.add("n");
		posOfInterest.add("v");
		posOfInterest.add("j");
		posOfInterest.add("r");
		LexicalStructureElementFilter elementFilter = new LexicalStructureElementFilter(stopwords, posOfInterest);

		TreeNodeFilter nodeFilter = new ContentBasedTreeNodeFilter(elementFilter);


		ExactMatchingStructureElementSimilarity exactMatching = new ExactMatchingStructureElementSimilarity(true);
		TreeNodeSimilarity contentNodeSimilarity = new ContentBasedTreeNodeSimilarity(exactMatching);

		TreePairRelTagger newTagger = new TreePairRelTagger(2, 0, "tree", nodeFilter, it.uniroma2.sag.kelp.data.manipulator.TreePairRelTagger.MARKING_POLICY.ON_NODE_LABEL, contentNodeSimilarity, 1);
		completeDataset.manipulate(newTagger);
		//END REL TAGGING


		StringLabel label = new StringLabel("Good"); //REMEMBER to change the label PotentiallyUseful into BAD in the input files!!

		System.out.println("----- TRAINING STATS: ");
		System.out.println("Good examples: " + trainset.getNumberOfPositiveExamples(label));
		System.out.println("not Good examples: " + trainset.getNumberOfNegativeExamples(label));
		System.out.println("total: " + trainset.getNumberOfExamples());
		
		System.out.println("----- DEV STATS: ");
		System.out.println("Good examples: " + devset.getNumberOfPositiveExamples(label));
		System.out.println("not Good examples: " + devset.getNumberOfNegativeExamples(label));
		System.out.println("total: " + devset.getNumberOfExamples());
		
		System.out.println("----- DEV STATS: ");
		System.out.println("Good examples: " + testset.getNumberOfPositiveExamples(label));
		System.out.println("not Good examples: " + testset.getNumberOfNegativeExamples(label));
		System.out.println("total: " + testset.getNumberOfExamples());

		//BEGIN KERNEL DEFINITION
		PartialTreeKernel ptk = new PartialTreeKernel(0.4f, 0.4f, 1, "tree");
		// ptk.setDeltaMatrix(new DynamicDeltaMatrix());
		//ptk.setMaxSubseqLeng(5);
		int cacheSize = getCacheSize(args, trainset, devset, testset);
		
		
		ptk.setSquaredNormCache(new DynamicIndexSquaredNormCache(cacheSize * 2));

		NormalizationKernel normKernel = new NormalizationKernel(ptk);
		UncrossedPairwiseProductKernel interpairKernel = new UncrossedPairwiseProductKernel(normKernel, false);

		LinearKernel kernelWSsimFeats = new LinearKernel("WSsim");
		LinearKernel kernelOnFeats = new LinearKernel("features");
		LinearKernel kernelOnThreadFeats = new LinearKernel("threadFeats");

		LinearKernelCombination comb = new LinearKernelCombination();
		comb.addKernel(1, kernelWSsimFeats);
		comb.addKernel(1, kernelOnFeats);
		comb.addKernel(1, kernelOnThreadFeats);
		comb.addKernel(1, interpairKernel);
		
		comb.setKernelCache(new DynamicIndexKernelCache(cacheSize));
		//END KERNEL DEFINITION

		BinaryCSvmClassification svm = new BinaryCSvmClassification(comb, label, c, c, false) {
			
			private Logger logger = LoggerFactory.getLogger(this.getClass());
			@Override
			protected void info(String msg) {
				super.info(msg);
				StringJoiner joiner = new StringJoiner(" ");
				for (int i=0;i<this.G.length; i++) {
					joiner.add(Float.toString(this.G[i]));
				}
				logger.info(joiner.toString());
			}
		};
		svm.setEps(0.003f);
		svm.learn(trainset);
		BinaryClassifier classifier = svm.getPredictionFunction();
		
		evaluateSet(args[0], "dev", devset, label, classifier);
		evaluateSet(args[0], "test", testset, label, classifier);
		System.out.println("time: " + Long.toString(System.currentTimeMillis() - start));
	}

	private static int getCacheSize(String[] args, SimpleDataset trainset, SimpleDataset devset, SimpleDataset testset) {
		if (args.length>5) {
			return Integer.parseInt(args[5]);
		}
		return trainset.getNumberOfExamples() + devset.getNumberOfExamples() + testset.getNumberOfExamples();
	}

	private static void evaluateSet(String classifierName, String datasetType, SimpleDataset dataset,
			StringLabel label, BinaryClassifier classifier)
			throws FileNotFoundException, UnsupportedEncodingException, IOException {
		//BEGIN PRINTING SCORES ON OUTPUT FILE + COLLECTING STATS FOR EVALUATION
		PrintWriter pw = new PrintWriter(classifierName + "-" + datasetType, "utf8");
		BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator(label);
		for(Example example : dataset.getExamples()){
			String id2 = example.getRepresentation("info").getTextFromData();
			String id1 = id2.split("_")[0] + "_" + id2.split("_")[1];
			BinaryMarginClassifierOutput output = classifier.predict(example);
			evaluator.addCount(example, output);
			Float score = output.getScore(label);
			boolean binaryDecision = (score>0);
			pw.println(getScorerLine(id1, id2, 0, score, binaryDecision));
		}
		pw.close();
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		serializer.writeValueOnFile(classifier, classifierName + "-classifier.klp");
		serializer.writeValueOnFile(classifier.getModel(), classifierName + "-model.klp");
		System.out.println("ACC: " + evaluator.getAccuracy());
		System.out.println("PREC: " + evaluator.getPrecision());
		System.out.println("REC: " + evaluator.getRecall());
		System.out.println("F1: " + evaluator.getF1());
		
	}
	
	public static String getScorerLine(String id1, String id2, int rank, float score, boolean binaryDecison){
		return String.join("\t", id1, id2, Integer.toString(rank), Float.toString(score), Boolean.toString(binaryDecison));
	}

}

