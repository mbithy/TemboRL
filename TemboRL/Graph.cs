using System;
using System.Collections.Generic;
using System.Text;

namespace TemboRL
{
    public class Graph
    {
        private bool NeedsBackPropagation = true;
        private List<Action> BackProp = new List<Action>();


        public Graph(bool needsBackProp)
        {
            NeedsBackPropagation = needsBackProp;
        }
        public Graph()
        {
            NeedsBackPropagation = true;
        }

        public void Backward()
        {
            for (var i = BackProp.Count - 1; i >= 0; i--)
            {
                BackProp[i](); // tick!
            }
        }

        public Matrix RowPluck(Matrix m, int ix)
        {
            CM.Assert(ix >= 0 && ix < m.Rows);
            var d = m.Columns;
            var output = new Matrix(d, 1);
            for (var i = 0; i < d; i++)
            {
				output.W[i] = m.W[d * ix + i];
            } // copy over the data
            if (NeedsBackPropagation)
            {
                var bvc= new Action(() =>{
                    for (var i = 0; i < d; i++)
                    {
                        m.DW[d * ix + i] += output.DW[i];
                    }
                });
                BackProp.Add(bvc);
            }
            return output;
        }

        public Matrix Tanh(Matrix m)
        {
            // tanh nonlinearity
            var output = new Matrix(m.Rows, m.Columns);
            var n = m.W.Length;
            for (var i = 0; i < n; i++)
            {
				output.W[i] = Math.Tanh(m.W[i]);
            }
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                 {
                     for (var i = 0; i < n; i++)
                     {
                         // grad for z = tanh(x) is (1 - z^2)
                         var mwi = output.W[i];
                         m.DW[i] += (1.0 - mwi * mwi) * output.DW[i];
                     }
                 });
                BackProp.Add(bvc);
            }
            return output;
        }

        public Matrix Sigmoid(Matrix m)
        {
            // sigmoid nonlinearity
            var output = new Matrix(m.Rows, m.Columns);
            var n = m.W.Length;
            for (var i = 0; i < n; i++)
            {
				output.W[i] = CM.Sig(m.W[i]);
            }
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                {
                    for (var i = 0; i < n; i++)
                    {
                        // grad for z = tanh(x) is (1 - z^2)
                        var mwi = output.W[i];
                        m.DW[i] += mwi * (1.0 - mwi) * output.DW[i];
                    }
                });
                BackProp.Add(bvc);
            }
            return output;
        }

        public Matrix Relu(Matrix m)
        {
            var output = new Matrix(m.Rows, m.Columns);
            var n = m.W.Length;
            for (var i = 0; i < n; i++)
            {
                output.W[i] = Math.Max(0, m.W[i]); // relu
            }
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                {
                    for (var i = 0; i < n; i++)
                    {
                        m.DW[i] += m.W[i] > 0 ? output.DW[i] : 0.0;
                    }
                });
                BackProp.Add(bvc);
            }
            return output;
        }
        public Matrix Multiply(Matrix m1, Matrix m2)
        {
            CM.Assert(m1.Columns == m2.Rows, "matrix multiplier dimensions misaligned");
            var n = m1.Rows;
            var d = m2.Columns;
            var output = new Matrix(n, d);
            for (var i = 0; i < m1.Rows; i++)
            { // loop over rows of m1
                for (var j = 0; j < m2.Columns; j++)
                { // loop over cols of m2
                    var dot = 0.0;
                    for (var k = 0; k < m1.Columns; k++)
                    { // dot product loop
                        dot += m1.W[m1.Columns * i + k] * m2.W[m2.Columns * k + j];
                    }
					output.W[d * i + j] = dot;
                }
            }
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                {
                    for (var i = 0; i < m1.Rows; i++)
                    { // loop over rows of m1
                        for (var j = 0; j < m2.Columns; j++)
                        { // loop over cols of m2
                            var dot = 0.0;
                            for (var k = 0; k < m1.Columns; k++)
                            { // dot product loop
                                dot += m1.W[m1.Columns * i + k] * m2.W[m2.Columns * k + j];
                            }
                            output.W[d * i + j] = dot;
                        }
                    }
                });
                BackProp.Add(bvc);
            }
            return output;
        }

        public Matrix Add(Matrix m1, Matrix m2)
        {
            CM.Assert(m1.W.Length == m2.W.Length);
            var output = new Matrix(m1.Rows, m1.Columns);
            for (var i = 0; i < m1.W.Length; i++)
            {
				output.W[i] = m1.W[i] + m2.W[i];
            }
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                {
                    for (var i = 0; i < m1.W.Length; i++)
                    {
                        output.W[i] = m1.W[i] + m2.W[i];
                    }
                });
                BackProp.Add(bvc);
            }
            return output;
        }

        public Matrix Dot(Matrix m1, Matrix m2)
        {
            CM.Assert(m1.W.Length == m2.W.Length);
            var output = new Matrix(1, 1);
            var dot = 0.0;
            for (var i = 0; i < m1.W.Length; i++)
            {
                dot += m1.W[i] * m2.W[i];
            }
            output.W[0] = dot;
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                {
                    for (var i = 0; i < m1.W.Length; i++)
                    {
                        m1.DW[i] += m2.W[i] * output.DW[0];
                        m2.DW[i] += m1.W[i] * output.DW[0];
                    }

                });
                BackProp.Add(bvc);
            }
            return output;
        }

        public Matrix Eltmul(Matrix m1, Matrix m2)
        {
            CM.Assert(m1.W.Length == m2.W.Length);
            var output = new Matrix(m1.Rows, m1.Columns);
            for (var i = 0; i < m1.W.Length; i++)
            {
				output.W[i] = m1.W[i] * m2.W[i];
            }
            if (NeedsBackPropagation)
            {
                var bvc = new Action(() =>
                {
                    for (var i = 0; i < m1.W.Length; i++)
                    {
                        m1.DW[i] += m2.DW[i] * output.DW[i];
                        m2.DW[i] += m1.W[i] * output.DW[i];
                    }
                });
                BackProp.Add(bvc);
            }
            return output;
        }

       
    }
}
