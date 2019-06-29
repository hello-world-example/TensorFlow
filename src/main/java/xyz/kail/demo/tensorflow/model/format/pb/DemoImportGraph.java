package xyz.kail.demo.tensorflow.model.format.pb;

import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;

/**
 * source  ~/Python/venv-tf/bin/activate
 * python train.py
 *
 * @author https://github.com/ZhuanZhiCode/TensorFlow-Java-Examples
 */
public class DemoImportGraph {

    public static void main(String[] args) throws IOException {

        try (Graph graph = new Graph()) {
            //导入图
            byte[] graphBytes = IOUtils.toByteArray(DemoImportGraph.class.getResourceAsStream("/models/invode01-model.pb"));
            graph.importGraphDef(graphBytes);

            //根据图建立Session
            try (Session session = new Session(graph)) {
                //相当于TensorFlow Python中的
                // sess.run(z, feed_dict = {'x': 10.0})
                float z = session.runner()
                        // 模型中定义的参数名，这里给 10.0F
                        .feed("x", Tensor.create(1.0F))
                        // 模型的中的结果，变量名
                        .fetch("z")
                        .run()
                        .get(0)
                        .floatValue();

                System.out.println("z = " + z);
            }
        }

    }
}