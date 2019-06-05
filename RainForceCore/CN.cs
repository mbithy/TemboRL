using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;

namespace TemboRL
{
    public static class CN
    {
        private static bool return_v = false;
        private static double v_val = 0.0f;
        public static string GetId()
        {
            return new Guid().ToString();
        }
        public static void Assert(bool condition, string message="Condition not true")
        {
            if (!condition)
            {
                throw new Exception(message);
            }
        }
        public static double[] ArrayOfZeros(this int size)
        {
            var array = new double[size];
            for(var i = 0; i < size; i++)
            {
                array[i] = 0.0f;
            }
            return array;
        }
        public static double[] ArrayOfZeros(this double[]array)
        {
            for (var i = 0; i < array.Length; i++)
            {
                array[i] = 0.0f;
            }
            return array;
        }

        public static double GaussRandom()
        {
            if (return_v)
            {
                return_v = false;
                return v_val;
            }
            var u = 2 * Random() - 1;
            var v = 2 * Random() - 1;
            var r = u * u + v * v;
            if (r == 0 || r > 1) return GaussRandom();
            //var c = Math.sqrt(-2 * Math.log(r) / r);
            double c = Math.Sqrt(-2 * Math.Log(r) / r);
            v_val = v * c; // cache this
            return_v = true;
            return u * c;
        }
        public static double RandF(double a, double b)
        {
            return Random() * (b - a) + a;
        }
        public static double RandI(double a, double b)
        {
            return Math.Floor(Random() * (b - a) + a);
        }
        public static double RandN(double mu,double std)
        {
            return mu + GaussRandom() * std;
        }
        private static double Random(double minimum = 0, double maximum = 1)
        {
            Random random = new Random();
            double nextDouble =random.NextDouble();
            return nextDouble * (maximum - minimum) + minimum;
        }

        public static void UpdateMatrix(Matrix m, double alpha)
        {
            var n = m.Rows * m.Columns;
            for (var i = 0; i < n; i++)
            {
                if (m.DW[i] != 0)
                {
                    m.W[i] += -alpha * m.DW[i];
                    m.DW[i] = 0;
                }
            }
        }

        public static void FillRandom(Matrix m, double mu, double std)
        {
            for (var i = 0; i < m.W.Length; i++)
            {
                m.W[i] = RandN(mu, std);
            }
        }

        public static Matrix RandomMatrix(int numberOfRows, int numberOfColumns, double mu, double std)
        {
            var m = new Matrix(numberOfRows, numberOfColumns);
            FillRandom(m, mu, std);
            return m;
        }

        public static void GradFillConst(Matrix m, double c)
        {
            for (var i = 0; i < m.DW.Length; i++)
            {
                m.DW[i] = c;
            }
        }

        public static double Sig(double x)
        {
            // helper function for computing sigmoid
            return 1.0 / (1 + Math.Exp(-x));
        }

        public static Matrix SoftMax(Matrix m)
        {
            var output = new Matrix(m.Rows, m.Columns); // probability volume
            var maxval = -999998.9999;
            for (var i = 0; i < m.W.Length; i++)
            {
                if (m.W[i] > maxval) maxval = m.W[i];
            }
            var s = 0.0;
            for (var i = 0; i < m.W.Length; i++)
            {
                output.W[i] = Math.Exp(m.W[i] - maxval);
                s += output.W[i];
            }
            for (var i = 0; i < m.W.Length; i++)
            {
                output.W[i] /= s;
            }
            // no backward pass here needed
            // since we will use the computed probabilities outside
            // to set gradients directly on m
            return output;
        }

        public static bool Contains(this List<Model>collection, Model m)
        {
            foreach(var model in collection)
            {
                if (model.Key == m.Key)
                {
                    return true;
                }
            }
            return false;
        }
        public static List<Model> InitLSTM(int inputSize, int[] hiddenSizes, int outputSize)
        {
            var models = new List<Model>();
            for (var d = 0; d < hiddenSizes.Length; d++)
            { // loop over depths
                var prev_size = d == 0 ? inputSize : hiddenSizes[d - 1];
                var hiddenSize = hiddenSizes[d];
                // gates parameters
                var key = "Wix" + d;
                var model = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model);
                key = "Wih" + d;
                var model2 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model2);
                key = "bi" + d;
                var model3 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model3);
                key = "Wfx" + d;
                var model4 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model4);
                key = "Wfh" + d;
                var model5 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model5);
                key = "bf" + d;
                var model6 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model6);
                key = "Wox" + d;
                var model7 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model7);
                key = "Woh" + d;
                var model8 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model8);
                key = "bo" + d;
                var model9 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model9);
                key = "Wcx" + d;
                var model10 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model10);
                key = "Wch" + d;
                var model11 = new Model(key, CN.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model11);
                key = "bc" + d;
                var model12 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model12);

            }
            // decoder params
            var model13 = new Model("Whd", CN.RandomMatrix(outputSize, hiddenSizes.LastOrDefault(), 0, 0.08));
            models.Add(model13);
            var model14 = new Model("bd", new Matrix(outputSize, 1));
            models.Add(model14);
            return models;
        }

        public static ForwardLSTModel ForwardLSTM(Graph G,List<Model>models, int[]hiddenSizes,Matrix x, ForwardLSTModel prev)
        {
            var hidden_prevs = new List<Matrix>();
            var cell_prevs = new List<Matrix>();
            // forward prop for a single tick of LSTM
            // G is graph to append ops to
            // model contains LSTM parameters
            // x is 1D column vector with observation
            // prev is a struct containing hidden and cell
            // from previous iteration
            if (prev == null || prev.H==null)
            {
                for (var d = 0; d < hiddenSizes.Length; d++)
                {
                    hidden_prevs.Add(new Matrix(hiddenSizes[d],1));
                    cell_prevs.Add(new Matrix(hiddenSizes[d], 1));
                }
            }
            else
            {
                hidden_prevs = prev.H;
                cell_prevs = prev.C;
            }

            var hidden = new List<Matrix>();
            var cell = new List<Matrix>();
            for (var d = 0; d < hiddenSizes.Length; d++)
            {
                var input_vector = d == 0 ? x : hidden[d - 1];
                var hidden_prev = hidden_prevs[d];
                var cell_prev = cell_prevs[d];
                // input gate
                var key = "Wix" + d;
                var h0 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);
                key = "Wih" + d;
                var h1 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);// G.mul(model[], hidden_prev);
                key = "bi" + d;
                var input_gate = Graph.Sigmoid(Graph.Add(Graph.Add(h0, h1), models.FirstOrDefault(m => m.Key == key).Value));// G.sigmoid(G.add(G.add(h0, h1), model[]));
                // forget gate
                key = "Wfx" + d;
                var h2 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);// G.mul(model[], input_vector);
                key = "Wfh" + d;
                var h3 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);// G.mul(model[], hidden_prev);
                key = "bf" + d;
                var forget_gate = Graph.Sigmoid(Graph.Add(Graph.Add(h2, h3), models.FirstOrDefault(m => m.Key == key).Value));// G.sigmoid(G.add(G.add(h2, h3), model[]));
                // output gate
                key = "Wox" + d;
                var h4 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);//G.mul(model[], input_vector);
                key = "Woh" + d;
                var h5 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);//G.mul(model[], hidden_prev);
                key = "bo" + d;
                var output_gate = Graph.Sigmoid(Graph.Add(Graph.Add(h4, h5), models.FirstOrDefault(m => m.Key == key).Value));// G.sigmoid(G.add(G.add(h4, h5), model[]));
                // write operation on cells
                key = "Wcx" + d;
                var h6 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);//G.mul(model[], input_vector);
                key = "Wch" + d;
                var h7 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);//G.mul(model[], hidden_prev);
                key = "bc" + d;
                var cell_write = Graph.Sigmoid(Graph.Add(Graph.Add(h6, h7), models.FirstOrDefault(m => m.Key == key).Value));// G.tanh(G.add(G.add(h6, h7), model[]));
                // compute new cell activation
                var retain_cell = Graph.Eltmul(forget_gate, cell_prev);// G.eltmul(forget_gate, cell_prev); // what do we keep from cell
                var write_cell = Graph.Eltmul(input_gate, cell_write);// G.eltmul(input_gate, cell_write); // what do we write to cell
                var cell_d = Graph.Add(retain_cell, write_cell);// G.add(retain_cell, write_cell); // new cell contents
                                                                // compute hidden state as gated, saturated cell activations
                var hidden_d = Graph.Eltmul(output_gate, Graph.Tanh(cell_d));// G.eltmul(output_gate, G.tanh(cell_d));
                hidden.Add(hidden_d);
                cell.Add(cell_d);
            }
            var model = models.FirstOrDefault(k => k.Key == "Whd");
            // one decoder to outputs at end
            var output = Graph.Add(Graph.Multiply(model.Value, hidden.LastOrDefault()), models.FirstOrDefault(k => k.Key == "Whd").Value);// G.add(G.mul(model["Whd"], hidden[hidden.length - 1]), model["bd"]);
            // return cell memory, hidden representation and output
            var ouu = new ForwardLSTModel
            {
                H = hidden,
                C = cell,
                O = output
            };
            return ouu;
        }
    }
}
