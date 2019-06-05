using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;

namespace TemboRL
{
    public class Solver
    {
        public double DecayRate = 0.999;
        public double SmoothEps = 1e-8;
        public List<Model> StepCache = new List<Model>();

        public dynamic Step(List<Model> model,int stepSize,double regc, double clipval)
        {
            dynamic solverStats = new ExpandoObject();
            var num_clipped = 0;
            var num_tot = 0;
            foreach (var k in model)
            {
                if (k.Value != null)
                {
                    var m = k.Value;

                    if (!CN.Contains(StepCache, k))
                    {
                        StepCache.Add(new Model(k.Key, new Matrix(k.Value.Rows, k.Value.Columns)));
                    }
                    var s = StepCache.FirstOrDefault(d => d.Key == k.Key);

                    for (var i = 0; i < m.W.Length; i++)
                    {
                        // rmsprop adaptive learning rate
                        var mdwi = m.DW[i];
                        s.Value.W[i] = s.Value.W[i] * DecayRate + (1.0 - DecayRate) * mdwi * mdwi;
                        // gradient clip
                        if (mdwi > clipval)
                        {
                            mdwi = clipval;
                            num_clipped++;
                        }
                        if (mdwi < -clipval)
                        {
                            mdwi = -clipval;
                            num_clipped++;
                        }
                        num_tot++;
                        // update (and regularize)
                        m.W[i] += -stepSize * mdwi / Math.Sqrt(s.Value.W[i] + SmoothEps) - regc * m.W[i];
                        m.DW[i] = 0; // reset gradients for next iteration
                    }
                }
            }
            solverStats.ratio_clipped = num_clipped * 1.0 / num_tot;
            return solverStats;
        }

        
    }

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

    public class ForwardLSTModel
    {
        public List<Matrix> H { get; set; }
        public List<Matrix> C { get; set; }
        public Matrix O { get; set; }
    }
}
