using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TemboRL.Models;
namespace TemboRL
{
    /// <summary>
    /// Thau shalt stay in memory until I say so
    /// </summary>
    public static class Tembo
    {
        private static bool return_v = false;
        private static double v_val = 0.0f;
        private static double nextId = 0;
        private static Graph Graph = new Graph();
        /// <summary>
        /// p+1 in memory is your next id
        /// </summary>
        /// <returns></returns>
        public static double NextId
        {
            get
            {
                nextId += 1;
                return nextId;
            }
        }

        /// <summary>
        /// GUID based
        /// </summary>
        /// <returns></returns>
        public static string GetId() => Guid.NewGuid().ToString();

        /// <summary>
        /// Fancy way to thow an exception influence by k.
        /// </summary>
        /// <param name="condition"></param>
        /// <param name="message"></param>
        public static void Assert(bool condition, string message = "Condition not true")
        {
            if (!condition)
            {
                throw new Exception(message);
            }
        }
        /*
         * pull request?
         * public static double Permuter(double n, double r)
        {
            var a=r+n-1!;
            var b=r!(n-1)!;
            return a / b;
        }*/
        public static double[] ArrayOfZeros(this int size)
        {
            var array = new double[size];
            for (var i = 0; i < size; i++)
            {
                array[i] = 0.0f;
            }
            return array;
        }
        public static double[] ArrayOfZeros(this double[] array)
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
        /// <summary>
        /// A javascript knock off
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static int RandomInt(double a, double b)
        {
            return Math.Floor(Random() * (b - a) + a).ToInt();
        }
        /// <summary>
        /// ->  mu + GaussRandom() * std
        /// </summary>
        /// <param name="mu"></param>
        /// <param name="std"></param>
        /// <returns></returns>
        public static double RandomNumber(double mu, double std)
        {
            return mu + GaussRandom() * std;
        }
        /// <summary>
        /// A javascript knock off
        /// </summary>
        /// <param name="minimum"></param>
        /// <param name="maximum"></param>
        /// <returns></returns>
        public static double Random(double minimum = 0, double maximum = 1)
        {
            Random random = new Random();
            double nextDouble = random.NextDouble();
            return nextDouble * (maximum - minimum) + minimum;
        }
        /// <summary>
        /// Converts a double to Int
        /// </summary>
        /// <param name="doub"></param>
        /// <returns></returns>
        public static int ToInt(this double doub) => int.Parse(doub.ToString(CultureInfo.InvariantCulture).Split('.')[0]);

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
        public static void UpdateNetwork(Dictionary<string,Matrix>net, double alpha)
        {
            foreach(var p in net)
            {
                UpdateMatrix(p.Value, alpha);
            }            
        }
        public static void NetZeroGrads(KeyValuePair<string, Matrix> net)
        {
            GradFillConst(net.Value, 0);
        }
        public static Matrix NetFlattenGrads(KeyValuePair<string, Matrix> net)
        {
            var g = new Matrix(net.Value.DW.Length, 1);
            var ix = 0;
            for (var i = 0; i < net.Value.DW.Length; i++)
            {
                g.W[ix] = net.Value.DW[i];
                ix++;
            }
            return g;
        }
        public static void FillRandom(Matrix m, double mu, double std)
        {
            for (var i = 0; i < m.W.Length; i++)
            {
                m.W[i] = RandomNumber(mu, std);
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
        public static bool Contains(this List<Model> collection, Model m)
        {
            foreach (var model in collection)
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
                var model = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model);
                key = "Wih" + d;
                var model2 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model2);
                key = "bi" + d;
                var model3 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model3);
                key = "Wfx" + d;
                var model4 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model4);
                key = "Wfh" + d;
                var model5 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model5);
                key = "bf" + d;
                var model6 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model6);
                key = "Wox" + d;
                var model7 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model7);
                key = "Woh" + d;
                var model8 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model8);
                key = "bo" + d;
                var model9 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model9);
                key = "Wcx" + d;
                var model10 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model10);
                key = "Wch" + d;
                var model11 = new Model(key, Tembo.RandomMatrix(hiddenSize, prev_size, 0, 0.08));
                models.Add(model11);
                key = "bc" + d;
                var model12 = new Model(key, new Matrix(hiddenSize, 1));
                models.Add(model12);
            }
            // decoder params
            var model13 = new Model("Whd", Tembo.RandomMatrix(outputSize, hiddenSizes.LastOrDefault(), 0, 0.08));
            models.Add(model13);
            var model14 = new Model("bd", new Matrix(outputSize, 1));
            models.Add(model14);
            return models;
        }
        public static ForwardLSTModel ForwardLSTM(Graph G, List<Model> models, int[] hiddenSizes, Matrix x, ForwardLSTModel prev)
        {
            var hidden_prevs = new List<Matrix>();
            var cell_prevs = new List<Matrix>();
            // forward prop for a single tick of LSTM
            // G is graph to append ops to
            // model contains LSTM parameters
            // x is 1D column vector with observation
            // prev is a struct containing hidden and cell
            // from previous iteration
            if (prev == null || prev.H == null)
            {
                for (var d = 0; d < hiddenSizes.Length; d++)
                {
                    hidden_prevs.Add(new Matrix(hiddenSizes[d], 1));
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
            //todo when key is null?
            for (var d = 0; d < hiddenSizes.Length; d++)
            {
                var input_vector = d == 0 ? x : hidden[d - 1];
                var hidden_prev = hidden_prevs[d];
                var cell_prev = cell_prevs[d];
                // input gate
                var key = "Wix" + d;
                var h0 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);
                key = "Wih" + d;
                var h1 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);
                key = "bi" + d;
                var input_gate = Graph.Sigmoid(Graph.Add(Graph.Add(h0, h1), models.FirstOrDefault(m => m.Key == key).Value));
                // forget gate
                key = "Wfx" + d;
                var h2 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);
                key = "Wfh" + d;
                var h3 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);
                key = "bf" + d;
                var forget_gate = Graph.Sigmoid(Graph.Add(Graph.Add(h2, h3), models.FirstOrDefault(m => m.Key == key).Value));
                // output gate
                key = "Wox" + d;
                var h4 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);
                key = "Woh" + d;
                var h5 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);
                key = "bo" + d;
                var output_gate = Graph.Sigmoid(Graph.Add(Graph.Add(h4, h5), models.FirstOrDefault(m => m.Key == key).Value));
                // write operation on cells
                key = "Wcx" + d;
                var h6 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, input_vector);
                key = "Wch" + d;
                var h7 = Graph.Multiply(models.FirstOrDefault(m => m.Key == key).Value, hidden_prev);
                key = "bc" + d;
                var cell_write = Graph.Sigmoid(Graph.Add(Graph.Add(h6, h7), models.FirstOrDefault(m => m.Key == key).Value));
                // compute new cell activation
                var retain_cell = Graph.Eltmul(forget_gate, cell_prev);
                var write_cell = Graph.Eltmul(input_gate, cell_write);
                var cell_d = Graph.Add(retain_cell, write_cell);
                var hidden_d = Graph.Eltmul(output_gate, Graph.Tanh(cell_d));
                hidden.Add(hidden_d);
                cell.Add(cell_d);
            }
            var model = models.FirstOrDefault(k => k.Key == "Whd");
            // one decoder to outputs at end
            var output = Graph.Add(Graph.Multiply(model.Value, hidden.LastOrDefault()), models.FirstOrDefault(k => k.Key == "Whd").Value);
            // return cell memory, hidden representation and output
            var ouu = new ForwardLSTModel
            {
                H = hidden,
                C = cell,
                O = output
            };
            return ouu;
        }
        public static int Maxi(double[] w)
        {
            // argmax of array w
            var maxv = w[0];
            var maxix = 0;
            for (var i = 1; i < w.Length; i++)
            {
                var v = w[i];
                if (v > maxv)
                {
                    maxix = i;
                    maxv = v;
                }
            }
            return maxix;
        }
        public static int SampleI(double[] w)
        {
            // sample argmax from w, assuming w are 
            // probabilities that sum to one
            var r = Tembo.RandomNumber(0, 1);
            var x = 0.0;
            var i = 0;
            while (true)
            {
                x += w[i];
                if (x > r)
                {
                    return i;
                }
                i++;
            }
            //wtf is this return doing here? idk! perhaps it was javascript thing but i'm leaving it for history
            return w.Length - 1; // pretty sure we should never get here?
        }
        public static void SetConst(double[] arr, double c)
        {
            for (var i = 0; i < arr.Length; i++)
            {
                arr[i] = c;
            }
        }
        public static int SampleWeighted(double[] p)
        {
            var r = Tembo.Random();
            var c = 0.0;
            for (var i = 0; i < p.Length; i++)
            {
                c += p[i];
                if (c >= r)
                {
                    return i;
                }
                //when update mode is sarsa wtf is reached
                //so just retun the largest
                if(c<=r && i >= p.Length - 1)
                {
                    return i;
                }
            }
            Tembo.Assert(false, "'wtf'");
            return 0;
        }
    }
}
