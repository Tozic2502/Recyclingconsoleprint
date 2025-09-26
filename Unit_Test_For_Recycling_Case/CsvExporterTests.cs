using Infrastructure;
using Microsoft.ML.Data;
using Xunit;
using System.IO;

public class CsvExporterTests
{
    [Fact]
    public void CsvExporter_CreatesFileWithHeader()
    {
        var path = "test_confusion.csv";
        var classes = new[] { "papir", "pap" };
        var cm = new ConfusionMatrix(2, new int[,] { { 1, 0 }, { 0, 1 } });

        CsvExporter.SaveConfusionMatrix(path, classes, cm);

        Assert.True(File.Exists(path));

        var content = File.ReadAllText(path);
        Assert.Contains("Actual/Predicted", content);
        Assert.Contains("papir", content);
    }
}

