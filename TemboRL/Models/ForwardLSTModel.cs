using System.Collections.Generic;
namespace TemboRL.Models
{
    public class ForwardLSTModel
    {
        public List<Matrix> H { get; set; }
        public List<Matrix> C { get; set; }
        public Matrix O { get; set; }
    }
}
