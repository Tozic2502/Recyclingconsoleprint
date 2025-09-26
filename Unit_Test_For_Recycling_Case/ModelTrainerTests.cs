using Infrastructure;
using Xunit;

public class ModelTrainerTests
{
    [Fact]
    public void TrainAndEvaluate_ShouldReturnMetrics()
    {
        var trainer = new ModelTrainer();
        var classes = new[] { "papir", "pap" };

        // For en rigtig test skal du have nogle små mapper med 1-2 billeder til papir/pap
        var (model, metrics) = trainer.TrainAndEvaluate(@"C:\DATA", classes);

        Assert.NotNull(model);
        Assert.True(metrics.MicroAccuracy >= 0);
    }
}
