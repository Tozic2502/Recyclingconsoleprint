using Domain;
using Infrastructure;
using Service;

namespace RecyclingConsolePrint
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "model", "model.pb");
            var labelsPath = Path.Combine(Environment.CurrentDirectory, "model", "labels.txt");

            var service = new TeachableMachinePredictionService(modelPath, labelsPath);

            //// Første gang: find navne i konsollen → kopier ind her
            //service.ConfigureTensors("serving_default_input_1:0", "StatefulPartitionedCall:0");

            //// Test
            //var image = new ImageData { ImagePath = "C:\\DATA\\testimage.jpg.jpg" };
            //var result = service.Predict(image);

            //Console.WriteLine($"📸 {result.ImagePath}");
            //Console.WriteLine($"🧾 Label: {result.Label}");
            //Console.WriteLine($"📊 Confidence: {result.Probability:P2}");


        }
    }
}
