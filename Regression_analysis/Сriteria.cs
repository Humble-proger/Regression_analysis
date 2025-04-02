
namespace Regression_analysis
{
    public class SREСriteria
    {
        private static double CalcRSS(Vectors calcTheta, Vectors matrixX, Vectors y) {
            var vector = y - (matrixX & calcTheta).T();
            vector *= vector;
            return Vectors.Sum(vector);
        }
        private static double CalcRSSH(Vectors y) {
            double result = 0.0;
            double avgY = Vectors.Sum(y)/y.Size;
            for (int i = 0; i < y.Size; i++) {
                result += Math.Pow(y[i] - avgY, 2);
            }
            return result;
        }
        public static double SRE(IModel model, Vectors calcTheta, Vectors x, Vectors y) {
            double rSS = CalcRSS(calcTheta, model.CreateMatrixX(x), y);
            double rSS_H = CalcRSSH(y);
            return (rSS_H - rSS) * (y.Size - model.CountRegressor) / (rSS * model.CountRegressor);
        }
    }
}