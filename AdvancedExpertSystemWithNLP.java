import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class AdvancedExpertSystemWithNLP {

    private Classifier classifier;
    private Tokenizer tokenizer;

    public AdvancedExpertSystemWithNLP() {
        classifier = new J48(); // يمكن استخدام أي نوع من أنواع الخوارزميات في Weka

        tokenizer = SimpleTokenizer.INSTANCE;
    }

    public void trainModel(Instances data) throws Exception {

        classifier.buildClassifier(data);
    }

    public String classifyPhone(String query) throws Exception {

        List<String> tokens = tokenize(query);


        double[] features = extractFeatures(tokens);

        Instance newPhone = createInstance(features);

        double classIndex = classifier.classifyInstance(newPhone);
        return newPhone.classAttribute().value((int) classIndex);
    }

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String[] words = tokenizer.tokenize(text);
        for (String word : words) {
            tokens.add(word.toLowerCase());
        }
        return tokens;
    }

    private double[] extractFeatures(List<String> tokens) {
        double processor = 0; // المعالج
        double memory = 0; // الذاكرة
        double camera = 0; // الكاميرا
        double screenType = 0; // نوع الشاشة
        double battery = 0; // البطارية
        double fastCharging = 0; // الشحن السريع
        double waterResistance = 0; // مقاومة الماء
        double screenSize = 0; // حجم الشاشة


        return new double[]{processor, memory, camera, screenType, battery, fastCharging, waterResistance, screenSize};
    }

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

        Instances data = new Instances("Phones", attributeList, 0);
        data.setClassIndex(data.numAttributes() - 1); // تحديد الفئة

        Instance instance = new DenseInstance(1.0, features);
        instance.setDataset(data);

        return instance;
    }

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        System.out.println("مرحبًا بك في نظام الخبير لاختيار الهواتف المحمولة");
        System.out.println("ما هو استفسارك؟");
        String query = scanner.nextLine();

        
        Instances data = buildDataset();

        
        AdvancedExpertSystemWithNLP expertSystem = new AdvancedExpertSystemWithNLP();
        expertSystem.trainModel(data);

        
        String recommendation = expertSystem.classifyPhone(query);
        System.out.println("يتم توصيلك بالهاتف ذو " + recommendation + " تكلفة.");

        scanner.close();
    }

    private static Instances buildDataset() {
        // إضافة عينات لمجموعة البيانات
        double[][] phoneData = {
            {4, 64, 12, 0, 3000, 1, 0, 5},
            {8, 128, 24, 1, 4000, 1, 1, 5.5},
            {12, 256, 48, 1, 5000, 1, 1, 6}
        };

        Attribute processor = new Attribute("processor");
        Attribute memory = new Attribute("memory");
        Attribute camera = new Attribute("camera");
        Attribute screenType = new Attribute("screenType");
        Attribute battery = new Attribute("battery");
        Attribute fastCharging = new Attribute("fastCharging");
        Attribute waterResistance = new Attribute("waterResistance");
        Attribute screenSize = new Attribute("screenSize");
        Attribute classAttribute = new Attribute("class", new String[]{"low", "medium", "high"});

        Instances data = new Instances("Phones",
                new ArrayList<>(List.of(processor, memory, camera, screenType, battery, fastCharging, waterResistance, screenSize, classAttribute)),
                0);
        data.setClassIndex(data.numAttributes() - 1);

        for (double[] features : phoneData) {
            Instance instance = new DenseInstance(1.0, features);
            instance.setDataset(data);
            data.add(instance);
        }

        return data;
    }
}
