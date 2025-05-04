using Avalonia.Data.Converters;
using System;

namespace RegressionApp.ViewModels
{
    public partial class MainWindowViewModel : ViewModelBase
    {
    }

    public class InverseBooleanConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value is bool boolValue)
                return !boolValue;
            return value; // если не bool, возвращаем как есть
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value is bool boolValue)
                return !boolValue;
            return value;
        }
    }
}
