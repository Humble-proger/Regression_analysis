

using System.Runtime.CompilerServices;
using System.Numerics;

namespace Regression_analysis
{
    abstract class Random_distribution
    {
        public readonly int Seed = 100;
        private Random rand;

        Random_distribution(int seed) {
            this.rand = new Random(seed);
            this.Seed = seed;
        }
        /*
        public double Uniform(double a, double b) => a + this.rand.NextDouble() * (b - a);
        public Vector2 Uniform(double a, double b, (int, int) Shape) {
            Vector2 vec = Vector2.Zeros(Shape);
            for (int i = 0; i < Shape.Item1; i++)
                for (int j = 0; j < Shape.Item2; j++)
                    vec[i][j] = this.Uniform(a, b);
        }
        */
    }
}
