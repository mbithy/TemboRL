namespace TemboRL.Models
{
    public class Model
    {
        public string Key { get; set; }
        public Matrix Value { get; set; }
        public Model(string key, Matrix m)
        {
            Key = key;
            Value = m;
        }
    }
}
