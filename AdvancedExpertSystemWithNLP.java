// made by @Taha Idra
// this Expert system used to help people to choose any phones that they wish, the code is very simple and clean, it is work with user input from kayboard.
// import nlp library to use word processor tools
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
// import weka library to use AI techniques in Java
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
// import ArrayList from util
import java.util.ArrayList;
import java.util.List;
// import Scanner function to get user input
import java.util.Scanner;
/*
start with definition class that call other private class to beginning System
*/
public class AdvancedExpertSystemWithNLP {
// Classifier & Tokenizer Class from opennlp 
    private Classifier classifier;
    private Tokenizer tokenizer;

    public AdvancedExpertSystemWithNLP() {
        classifier = new J48();

        tokenizer = SimpleTokenizer.INSTANCE;
    }//end AdvancedExpertSystemWithNLP()

    public void trainModel(Instances data) throws Exception {

        classifier.buildClassifier(data);
    }//end trainModel

    public String classifyPhone(String query) throws Exception {

        List<String> tokens = tokenize(query);


        double[] features = extractFeatures(tokens);

        Instance newPhone = createInstance(features);

        double classIndex = classifier.classifyInstance(newPhone);
        return newPhone.classAttribute().value((int) classIndex);
    }//end classifyPhone

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String[] words = tokenizer.tokenize(text);
        for (String word : words) {
            tokens.add(word.toLowerCase());
        }//end for
        return tokens;
    }//end tokenize

    private double[] extractFeatures(List<String> tokens) {
        double processor = 0; // processor variable
        double memory = 0;  // memory variable
        double camera = 0; // camera variable
        double screenType = 0; // screenType variable
        double battery = 0; // battery variable
        double fastCharging = 0; // fastCharging variable
        double waterResistance = 0; // waterResistance variable
        double screenSize = 0; // screenSize variable


        return new double[]{processor, memory, camera, screenType, battery, fastCharging, waterResistance, screenSize};
    }//end extractFeatures

    private Instance createInstance(double[] features) {
        
        Attribute processor = new Attribute("processor");
        Attribute memory = new Attribute("memory");
        Attribute camera = new Attribute("camera");
        Attribute screenType = new Attribute("screenType");
        Attribute battery = new Attribute("battery");
        Attribute fastCharging = new Attribute("fastCharging");
        Attribute waterResistance = new Attribute("waterResistance");
        Attribute screenSize = new Attribute("screenSize");
        Attribute classAttribute = new Attribute("class", new String[]{"low", "medium", "high"});
// Add attributes to the attributeList
        ArrayList<Attribute> attributeList = new ArrayList<>();
        attributeList.add(processor);
        attributeList.add(memory);
        attributeList.add(camera);
        attributeList.add(screenType);
        attributeList.add(battery);
        attributeList.add(fastCharging);
        attributeList.add(waterResistance);
        attributeList.add(screenSize);
        attributeList.add(classAttribute);
        
// Create an object named "Phones" with the attributeList
        Instances data = new Instances("Phones", attributeList, 0);
        data.setClassIndex(data.numAttributes() - 1); // To set the class index

        Instance instance = new DenseInstance(1.0, features);
        instance.setDataset(data);

        return instance;
    }//end createInstance

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        System.out.println("أهلين وسهلين في النظام الخبير المخصص للمساعدة في اختيار هاتفك");
        System.out.println("كيف نقدر نساعدك");
        String query = scanner.nextLine();

// Set the dataset of the instance to the created Instances object
        Instances data = buildDataset();

        
        AdvancedExpertSystemWithNLP expertSystem = new AdvancedExpertSystemWithNLP();
        expertSystem.trainModel(data);

        
        String recommendation = expertSystem.classifyPhone(query);
        System.out.println("يتم توصيلك بالهاتف ذو " + recommendation + " تكلفة.");

        scanner.close();
    }// end main()

// Define phone data including processor, memory, camera, screen type, battery, fast charging, water resistance, and screen size
// Each array within phoneData represents a phone, with values in the order of processor, memory, camera, screen type, battery, fast charging, water resistance, and screen size
   private static Instances buildDataset() {
       
        double[][] phoneData = {
            {4, 64, 12, 0, 3000, 1, 0, 5},
            {8, 128, 24, 1, 4000, 1, 1, 5.5},
            {12, 256, 48, 1, 5000, 1, 1, 6}
        };
       
// Create attribute for processor, memory, camera, screenType, battery, fastCharging, waterResistance, screenSize, classAttribute
        Attribute processor = new Attribute("processor");
        Attribute memory = new Attribute("memory");
        Attribute camera = new Attribute("camera");
        Attribute screenType = new Attribute("screenType");
        Attribute battery = new Attribute("battery");
        Attribute fastCharging = new Attribute("fastCharging");
        Attribute waterResistance = new Attribute("waterResistance");
        Attribute screenSize = new Attribute("screenSize");
        Attribute classAttribute = new Attribute("class", new String[]{"low", "medium", "high"});
        
// Create Instances object with the name (Phones) to hold phone data
// Initialize the Instances object with a list of attributes including processor, memory, camera, screenType, battery, fastCharging, waterResistance, screenSize, and aslo classAttribute
// Set object to 0 initially
        Instances data = new Instances("Phones",
                new ArrayList<>(List.of(processor, memory, camera, screenType, battery, fastCharging, waterResistance, screenSize, classAttribute)),
                0);
        data.setClassIndex(data.numAttributes() - 1);

        for (double[] features : phoneData) {
            Instance instance = new DenseInstance(1.0, features);
            instance.setDataset(data);
            data.add(instance);
        }//end for

        return data;
    }//end buildDataset()
}// end main
