

public class WasteSortingTest
    {
        [Fact]
        public void ImageData_ShouldStorePathAndLabel()
        {
            var data = new ImageData { ImagePath = "test.jpg", Label = "papir" };

            Assert.Equal("test.jpg", data.ImagePath);
            Assert.Equal("papir", data.Label);
        }
    }

