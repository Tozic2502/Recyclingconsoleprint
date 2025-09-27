using Microsoft.ML.Data;

namespace Domain;

public class ImageData
{
    public string ImagePath { get; set; } = "";
    public string Label { get; set; } = "";
    public byte[] Image { get; set; } 
}
