namespace TemboRL.Models
{
    public class Matrix
    {
        /// <summary>
        /// N
        /// </summary>
        public int Rows { get; }
        /// <summary>
        /// D
        /// </summary>
        public int Columns { get; set; }
        public double[] W { get; set; }
        public double [] DW { get; set; }
        /*
         * This way we can go straight to
         * a matrix without looping over
         */
        public string Id { get; private set; }
        public Matrix(int numberOfRows, int numberOfColumns, string id="")
        {
            Rows = numberOfRows;
            Columns = numberOfColumns;
            W = Tembo.ArrayOfZeros(Rows * numberOfColumns);
            DW= Tembo.ArrayOfZeros(Rows * numberOfColumns);
            Id = id != "" ? id : Tembo.GetId();
        }
        public double Get(int row, int col)
        {
            var ix = (Columns * row) + col;
            Tembo.Assert(ix >= 0 && ix < W.Length);
            return W[ix];
        }
        public void Set(int row, int col, double value)
        {
            var ix = (Columns * row) + col;
            Tembo.Assert(ix >= 0 && ix < W.Length);
            W[ix] = value;
        }
        public void Set(double[] arr)
        {
            //W = new double[arr.Length];
            for (var i = 0; i < arr.Length; i++)
            {
                W[i] = arr[i];
            }
        }
        public void SetColumn(Matrix m,int i)
        {
            for (var q = 0; q < m.W.Length; q++)
            {
                W[(Columns * q) + i] = m.W[q];
            }
        }
    }
}
