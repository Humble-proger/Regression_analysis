using Regression_analysis.Types;

namespace Regression_analysis.Regretion
{
    public interface IModel
    {
        public Vectors VectorFunc(Vectors x);
        public double True_value(Vectors x);
        public Vectors CreateMatrixX(Vectors x);
    }

    public class LiniarModel : IModel
    {
        public readonly bool FreeMember = false;
        public readonly int CountFacts = 1;
        public readonly int CountRegressor = 1;
        public readonly Vectors TrueTheta;
        public readonly (int, int)[] RelatedFacts;

        public LiniarModel(int count_facts, (int, int)[] related_facts, Vectors truetheta, bool free_member = false) {
            CountFacts = count_facts;
            FreeMember = free_member;
            // Фильтр для связанных фактов
            int left, right;
            for (int i = 0; i < related_facts.Length; i++)
            {
                (left, right) = related_facts[i];
                if (left > CountFacts || right > CountFacts)
                    throw new Exception("Inconsistency between the number of facts and the number of the related fact.");
                if (left > right)
                    related_facts[i] = (right, left);
            }
            RelatedFacts = related_facts.Distinct().ToArray();
            // Считаем количество регрессор
            CountRegressor = count_facts;
            if (free_member)
                CountRegressor += 1;
            CountRegressor += RelatedFacts.Length;
            if (truetheta.Shape.Item2 == 1)
                truetheta = truetheta.T();
            else if (truetheta.Shape.Item1 != 1)
                throw new Exception("Vector truetheta cannot be a matrix.");
            if (truetheta.Size != CountRegressor)
                throw new Exception("Incorrect input of true theta values.");
            TrueTheta = truetheta;
        }
        public Vectors VectorFunc(Vectors x) {
            if (x.Shape.Item2 == 1)
                x = x.T();
            else if (x.Shape.Item1 != 1)
                throw new Exception("Vector x cannot be a matrix.");
            if (x.Size != CountFacts) throw new Exception($"The size of the vector x does not match CountFacts. Number of facts: {CountFacts}. Size of x: {x.Size}.");
            double[] result = new double[CountRegressor];
            int index = 0;
            if (FreeMember)
                result[++index] = 1;
            for (int i = 0; i < CountFacts; i++, index++)
                result[index] = x[0, i];
            for (int i = 0; i < RelatedFacts.Length; i++, index++)
                result[index] = x[0, RelatedFacts[i].Item1] * x[0, RelatedFacts[i].Item2];
            return new Vectors(result);
        }
        public double True_value(Vectors x) {
            if (x.Shape.Item2 == 1)
                x = x.T();
            else if (x.Shape.Item1 != 1)
                throw new Exception("Vector x cannot be a matrix.");
            if (x.Size != CountFacts) throw new Exception($"The size of the vector x does not match CountFacts. Number of facts: {CountFacts}. Size of x: {x.Size}.");
            int index = 0;
            double result = 0;
            if (FreeMember)
                result += TrueTheta[0, ++index];
            for (int i = 0; i < CountFacts; i++, index++)
                result += x[0, i] * TrueTheta[0, index];
            for (int i = 0; i < RelatedFacts.Length; i++, index++)
                result += x[0, RelatedFacts[i].Item1] * x[0, RelatedFacts[i].Item2] * TrueTheta[0, index];
            return result;
        }
        public Vectors CreateMatrixX(Vectors x) {
            if (x.Shape.Item2 != CountFacts)
                throw new Exception($"Incorrect matrix size x.The matrix row size is {x.Shape.Item2}, but should be {CountFacts}");
            Vectors Result = Vectors.InitVectors((x.Shape.Item1, CountRegressor));
            for (int i = 0; i < x.Shape.Item1; i++) {
                Result.SetRow(VectorFunc(Vectors.GetRow(x, i)), i);
            }
            return Result;
        }
    }
}