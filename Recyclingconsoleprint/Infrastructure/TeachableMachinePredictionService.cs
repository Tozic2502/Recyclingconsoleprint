using System;
using System.Collections.Generic;
using System.Drawing; // Husk: kræver System.Drawing.Common
using System.IO;
using System.Linq;
using Domain;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding; // Gør at du kan bruge 'tf'

namespace Infrastructure
{
    public class TeachableMachinePredictionService
    {
        private readonly string _modelPath;
        private readonly string[] _labels;
        private Session? _session;
        private string _inputName = "";
        private string _outputName = "";

        public TeachableMachinePredictionService(string modelPath, string labelsPath)
        {
            _modelPath = modelPath;
            _labels = File.ReadAllLines(labelsPath);
            LoadModel();
        }

        private void LoadModel()
        {
            Console.WriteLine("🧠 Indlæser TensorFlow model...");
            tf.compat.v1.disable_eager_execution(); // Kræves til graf/session-style kørsel

            var graph = new Graph().as_default();
            var bytes = File.ReadAllBytes(_modelPath);


_session = tf.Session(graph);

            Console.WriteLine("✅ Model indlæst.");

            // Udskriv alle tensorer for at finde input/output navne første gang
            Console.WriteLine("📜 Tensorer i grafen:");
            foreach (var op in graph.get_operations())
                Console.WriteLine($" - {op.name}");

            Console.WriteLine("\n👉 Find dit input/output tensor-navn ovenfor (fx 'serving_default_input_1:0')");
        }

        /// <summary>
        /// Sæt input/output tensor-navne efter at du har udskrevet dem i konsollen
        /// </summary>
        public void ConfigureTensors(string inputName, string outputName)
        {
            _inputName = inputName;
            _outputName = outputName;
        }

        public PredictionResult Predict(ImageData image)
        {
            if (string.IsNullOrWhiteSpace(_inputName) || string.IsNullOrWhiteSpace(_outputName))
                throw new InvalidOperationException("⚠️ Du skal først sætte input/output tensor-navne via ConfigureTensors().");

            // Check if the image file exists
            if (!File.Exists(image.ImagePath))
                throw new FileNotFoundException($"Image file not found: {image.ImagePath}");

            // Try to load the image and catch invalid format errors
            Bitmap bmp;
            try
            {
                bmp = new Bitmap(image.ImagePath);
            }
            catch (Exception ex)
            {
                throw new ArgumentException($"Failed to load image. Ensure the file is a valid image format. Details: {ex.Message}", ex);
            }

            using (bmp)
            using (var resized = new Bitmap(bmp, new Size(224, 224)))
            {
                var inputData = new float[1, 224, 224, 3];
                for (int y = 0; y < 224; y++)
                {
                    for (int x = 0; x < 224; x++)
                    {
                        var pixel = resized.GetPixel(x, y);
                        inputData[0, y, x, 0] = pixel.R / 255f;
                        inputData[0, y, x, 1] = pixel.G / 255f;
                        inputData[0, y, x, 2] = pixel.B / 255f;
                    }
                }

                var tensor = np.array(inputData);

                var resultTensors = _session.run(
                    fetches: new[] { _outputName },
                    new FeedItem(_inputName, tensor)
                );

                var resultTensor = resultTensors[0].ToArray<float>();
                int maxIndex = Array.IndexOf(resultTensor, resultTensor.Max());

                return new PredictionResult
                {
                    ImagePath = image.ImagePath,
                    Label = _labels[maxIndex],
                    Probability = resultTensor[maxIndex]
                };
            }
        }
    }
}
