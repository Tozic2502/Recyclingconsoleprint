using Domain;
using System.Globalization;
using System.Text;

namespace Infrastructure
{
    public class CsvExporter
    {
        public void ExportToCsv(IEnumerable<PredictionResult> results, string outputPath)
        {
            var sb = new StringBuilder();
            sb.AppendLine("ImagePath,Label,Probability");

            foreach (var result in results)
            {
                sb.AppendLine($"{result.ImagePath},{result.Label},{result.Probability.ToString("F4", CultureInfo.InvariantCulture)}");
            }

            File.WriteAllText(outputPath, sb.ToString(), Encoding.UTF8);
        }
    }
}
