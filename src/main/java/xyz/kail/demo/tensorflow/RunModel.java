//package xyz.kail.demo.tensorflow;
//
//import org.tensorflow.Graph;
//import org.tensorflow.Session;
//import org.tensorflow.Tensor;
//
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.Arrays;
//
//public class RunModel {
//
//    public void predict() throws Exception {
//        try (Graph graph = new Graph()) {
//            graph.importGraphDef(Files.readAllBytes(Paths.get(
//                    "models/model.pb"
//            )));
//
//            try (Session sess = new Session(graph)) {
//                // 自己构造一个输入
//                float[][] input = {
//                        {56, 632, 675, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
//                };
//                try (Tensor x = Tensor.create(input);
//                     // input是输入的name，output是输出的name
//                     Tensor y = sess.runner()
//                             .feed("input", x)
//                             .fetch("output")
//                             .run().get(0)
//                ) {
//
//                    float[][] result = new float[1][y.shape()[1]];
//                    y.copyTo(result);
//                    System.out.println(Arrays.toString(y.shape()));
//                    System.out.println(Arrays.toString(result[0]));
//                }
//            }
//        }
//    }
//}
