import pickle
import matplotlib
matplotlib.use('Agg')  # Не использовать GUI
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet
from .models import train_model
from rest_framework.decorators import api_view
from .serializers import CountrySerializer

with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

class PredictionViewSet(ViewSet):
    @action(detail=False, methods=['get'])
    def predict(self, request):
        country = request.GET.get('country', '')
        
        data = {
            'new_cases': [100],
            'new_deaths': [2],
            'total_cases': [1000],
            'total_deaths': [50],
            'Rt': [1.2]
        }
        
        df_new = pd.DataFrame(data)

        X_new_scaled = scaler.transform(df_new)

        prediction = clf.predict(X_new_scaled)

        safety_levels = {
            0: 'No danger',
            1: 'Follow precautions',
            2: 'Better not to travel'
        }

        return JsonResponse({'prediction': safety_levels.get(prediction[0], 'Unknown')})

    @action(detail=False, methods=['get'], url_path='graphic')
    def graphic(self, request):
        df = pd.read_csv('Preprocessed_Data.csv')

        if 'date' not in df.columns or 'new_cases' not in df.columns:
            return HttpResponse("Ошибка: В CSV-файле нет нужных колонок.", status=400)

        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['new_cases'], label='Новые случаи')
        plt.title('Динамика заражений')
        plt.xlabel('Дата')
        plt.ylabel('Число новых случаев')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format="png")
        plt.close() 

        img_buffer.seek(0)

        return HttpResponse(img_buffer.getvalue(), content_type="image/png")




@api_view(['POST'])
def predict_by_post(request):
    serializer = CountrySerializer(data=request.data)

    if serializer.is_valid():
        country = serializer.validated_data['country']

        df_selected = pd.read_csv('Preprocessed_Data.csv')
        country_data = df_selected[df_selected['location'] == country]

        if country_data.empty:
            return Response({"error": f"No data found for country: {country}"}, status=404)

        latest_data = country_data.sort_values(by='date', ascending=False).iloc[0]

        df_new = pd.DataFrame([{
            'new_cases': latest_data['new_cases'],
            'new_deaths': latest_data['new_deaths'],
            'total_cases': latest_data['total_cases'],
            'total_deaths': latest_data['total_deaths'],
            'Rt': latest_data['Rt']
        }])

        X_new_scaled = scaler.transform(df_new)
        prediction = clf.predict(X_new_scaled)

        safety_levels = {
            0: 'No danger',
            1: 'Follow precautions',
            2: 'Better not to travel'
        }

        return Response({
            'country': country,
            'prediction': safety_levels.get(prediction[0], 'Unknown')
        })
    
    return Response(serializer.errors, status=400)
