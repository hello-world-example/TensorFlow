package xyz.kail.demo.tensorflow;

import org.tensorflow.*;

import java.nio.charset.StandardCharsets;

public class LoadPriceModel {

    private static Tensor newStringTensor(String str) {
        byte[][] data = new byte[1][];
        data[0] = str.getBytes(StandardCharsets.UTF_8);

        return Tensors.create(data);


//        return Tensors.create(str.getBytes(StandardCharsets.UTF_8));
//        return Tensor.create(String.class, new long[]{1}, ByteBuffer.wrap(str.getBytes(StandardCharsets.UTF_8)));
    }

    private static Tensor newDoubleTensor(double d) {
        return Tensor.create(new double[]{d});
//        return Tensor.create(d);
    }

    private static Tensor newFloatTensor(float d) {
        return Tensor.create(new float[]{d});
//        return Tensor.create(d);
    }

    public static void main(String[] args) {
        // 打印 TensorFlow 版本 (1.12.0)
        System.out.println(TensorFlow.version());

        // saved model 文件路径
        String path = "/Users/kail/Desktop/model_price/1548654250/";

        // 加载 模型文件
        try (SavedModelBundle b = SavedModelBundle.load(path, "serve")) {

            // 创建 Session
            try (Session sess = b.session()) {
//                Tensor.create()

//                ByteBuffer.wrap(Tensor.create(String.class, new long[1])
//                Tensor.create(String.class, new long[1],

                // 运行模型，获取结果 14.0f.
                Tensor<?> tensor = sess.runner()
                        // 投喂 模型参数 x
                        // brand
                        .feed("brand", newStringTensor("现代"))
                        // family
                        .feed("family", newStringTensor("朗动"))
                        // model
                        .feed("model", newStringTensor("2015款 1.6L 自动领先型"))
                        // vehicle_type
                        .feed("vehicle_type", newStringTensor("轿车"))
                        // use_nature
                        .feed("use_nature", newStringTensor("非营运"))
                        // license_nature
                        .feed("license_nature", newStringTensor("私牌"))
                        // turbo
                        .feed("turbo", newStringTensor("自然吸气"))
                        // gear
                        .feed("gear", newStringTensor("手自一体"))
                        // emission
                        .feed("emission", newStringTensor("国IV(国V)"))
                        // license_prefix
                        .feed("license_prefix", newStringTensor("粤"))
                        // years
                        .feed("years", newDoubleTensor(3.3D))
                        // model_years
                        .feed("model_years", newDoubleTensor(3.64D))
                        // new_car_price
                        .feed("new_car_price", newDoubleTensor(12.78D))
                        // distance
                        .feed("distance", newDoubleTensor(106809D))
                        // exhaust
                        .feed("exhaust", newDoubleTensor(1.6D))
                        // transfer_number
                        .feed("transfer_number", newDoubleTensor(0D))
                        // equipment
                        .feed("equipment", newFloatTensor(0F))
                        // exterior
                        .feed("exterior", newFloatTensor(12F))
                        // interior
                        .feed("interior", newFloatTensor(3F))
                        // skeleton
                        .feed("skeleton", newFloatTensor(2F))
                        // 获取模型计算的结果 predictions
                        .fetch("add:0")
                        .run()
                        .get(0);

                // 打印结果
                System.out.println(tensor.floatValue());
            }
        }
    }

}
