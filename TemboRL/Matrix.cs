using System;
using System.Collections.Generic;
using System.Text;

namespace TemboRL
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
        public string Id { get; private set; }
        public Matrix(int numberOfRows, int numberOfColumns, string id="")
        {
            Rows = numberOfRows;
            Columns = numberOfColumns;
            W = CM.ArrayOfZeros(Rows * numberOfColumns);
            DW= CM.ArrayOfZeros(Rows * numberOfColumns);
            Id = id != "" ? id : new Guid().ToString();
        }

        public double Get(int row, int col)
        {
            var ix = (Columns * row) + col;
            CM.Assert(ix >= 0 && ix < W.Length);
            return W[ix];
        }

        public void Set(int row, int col, double value)
        {
            var ix = (Columns * row) + col;
            CM.Assert(ix >= 0 && ix < W.Length);
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
