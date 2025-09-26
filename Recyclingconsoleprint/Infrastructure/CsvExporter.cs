using Microsoft.ML.Data;
using System.IO;

namespace Infrastructure;

public static class CsvExporter
{
    public static void SaveConfusionMatrix(string path, string[] classes, ConfusionMatrix cm)
    {
        using (var writer = new StreamWriter(path))
        {
            writer.Write("Actual/Predicted");
            foreach (var cls in classes)
                writer.Write($",{cls}");
            writer.WriteLine();

            for (int i = 0; i < cm.Counts.Count; i++)
            {
                writer.Write(classes[i]);
                for (int j = 0; j < cm.Counts[i].Count; j++)
                {
                    writer.Write($",{cm.Counts[i][j]}");
                }
                writer.WriteLine();
            }
        }
    }
}
