import torch
import torch.nn as nn

class Conv_3_k(nn.Module):
    ''' Convolución 2D con kernel 3x3, padding 1 y stride 1
        mantenemos constantes las dimensiones espaciales
    '''

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        return self.conv1(x)
    
class DoubleConv(nn.Module):
    ''' Bloque de dos convoluciones con ReLU y BatchNorm'''

    def __init__(self, channels_in, channels_out,channels_time=None):
        super().__init__()
        # 1. Definición de la inyección de tiempo
        self.time_projection = None
        if channels_time is not None:
            # Proyecta el vector de tiempo (channels_time) al tamaño de la salida (channels_out)
            self.time_projection = nn.Linear(channels_time, channels_out)

        # Primer Bloque (hasta aquí la imagen tiene channels_out)
        self.first_block = nn.Sequential(
            Conv_3_k(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

        # Segundo Bloque
        self.second_block = nn.Sequential(
            Conv_3_k(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )


    def forward(self, x,time_emb=None):

        # 1. Ejecutar el Primer Bloque (3 canales -> 64 canales)
        x = self.first_block(x) # x ahora tiene (B, channels_out, H, W)
        # 2. Inyección del Tiempo (CORREGIDA: Ahora los canales de x y emb coinciden)
        if self.time_projection is not None and time_emb is not None:
            emb = self.time_projection(time_emb) 
            
            # Sumamos el embedding a las activaciones de 64 canales
            x = x + emb.view(emb.size(0), emb.size(1), 1, 1)
            # 3. Ejecutar el Segundo Bloque (64 canales -> 64 canales)
        x = self.second_block(x)
        return x
        
class Down_Conv(nn.Module):
    ''' Bloque de Downsampling + DoubleConv y pasa time_emb.'''
    def __init__(self, channels_in, channels_out,channels_time):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(channels_in, channels_out, channels_time=channels_time)
        

    def forward(self, x, time_emb):
        x = self.maxpool(x)
        return self.conv(x, time_emb)

class Up_Conv(nn.Module):
    ''' Bloque de Upsampling + DoubleConv utilizando ConvTranspose2d y pasa time_emb.'''
    def __init__(self, channels_in, channels_out,channels_time,use_output_padding=False):
        super().__init__()

        # El número de canales se reduce a la mitad en la subida
        new_channels_in = channels_in//2
        # Capa de Transposición Convolucional
        # Usamos stride=2 para subir el tamaño (ej. 12 -> 24)
        # Si use_output_padding es True (para dimensiones impares como 25/50), 
        # forzamos la salida a ser tamaño + 1 (ej. 24 -> 25)
        self.upsample_layer = nn.ConvTranspose2d(channels_in, new_channels_in, 
                                                  kernel_size=2, 
                                                  stride=2, 
                                                  output_padding=1 if use_output_padding else 0)
        # La capa decoder ahora tiene el doble de canales de entrada
        # (new_channels_in * 2) porque concatena el skip connection (channels_out) 
        # con la salida del upsample (channels_out)
        
        self.decoder = DoubleConv(channels_in, channels_out,channels_time=channels_time)

    def forward(self, x1, x2,time_emb):
        '''
        x1: entrada a upsamplear
        x2: entrada para concatenar
        '''
        x1 = self.upsample_layer(x1)
        #Corregir desajuste si x1 y x2 son diferentes después del ConvTranspose2d
        # (Esto es redundante si output_padding está correcto, pero actúa como red de seguridad)
        if x1.shape[2] != x2.shape[2] or x1.shape[3] != x2.shape[3]:
             x1 = nn.functional.interpolate(x1, size=x2.shape[2:], mode='nearest')
        x = torch.cat([x2, x1], dim=1)  # Concatenamos en el canal
        return self.decoder(x,time_emb)

class UNET(nn.Module):
    '''U-Net model'''
    def __init__(self, channels_in, channels, channels_out):
        super().__init__()
        # Definimos el tamaño del embedding de tiempo (usamos 'channels' como base)
        CHANNELS_TIME = channels
        # --- Nuevo: Time Embedding ---
        self.time_embed = nn.Sequential(
            # Típicamente, t se convierte en un vector largo (ej. 4 * channels)
            nn.Linear(1, channels * 4), 
            nn.ReLU(inplace=True),
            # Y luego en un vector más pequeño para inyectar en las capas
            nn.Linear(channels * 4, CHANNELS_TIME) 
        )
        # -----------------------------

        self.inconv = DoubleConv(channels_in , channels, channels_time= CHANNELS_TIME) # 64, 100, 100
        self.down1 = Down_Conv(channels, channels*2,CHANNELS_TIME) # 128, 50, 50
        self.down2 = Down_Conv(channels*2, channels*4, CHANNELS_TIME) # 256, 25, 25
        self.down3 = Down_Conv(channels*4, channels*8, CHANNELS_TIME) # 512, 12, 12

        self.down4 = Down_Conv(channels*8, channels*16, CHANNELS_TIME) # 1024, 6, 6

        self.up1 = Up_Conv(channels*16, channels*8,CHANNELS_TIME, use_output_padding=True) # 512, 12, 12
        self.up2 = Up_Conv(channels*8, channels*4,CHANNELS_TIME, use_output_padding=True) # 256, 25, 25
        self.up3 = Up_Conv(channels*4, channels*2,CHANNELS_TIME) # 128, 50, 50
        self.up4 = Up_Conv(channels*2, channels,CHANNELS_TIME) # 64, 100, 100

        self.outconv = nn.Conv2d(channels, channels_out, kernel_size=1, stride=1) # 1, 100, 100

    def forward(self, x, t):

        # 1. Calcular Time Embedding
        t = t.unsqueeze(-1) # t: (B) -> (B, 1) para la capa Linear
        time_emb = self.time_embed(t) # time_emb: (B, channels)

        # 2. Inyectar en la primera capa
        # DEBES modificar inconv para aceptar time_emb
        x1 = self.inconv(x, time_emb)
        x2 = self.down1(x1, time_emb)
        x3 = self.down2(x2, time_emb)
        x4 = self.down3(x3, time_emb)
        x5 = self.down4(x4, time_emb)

        u1 = self.up1(x5, x4, time_emb)
        u2 = self.up2(u1, x3, time_emb)
        u3 = self.up3(u2, x2, time_emb)
        u4 = self.up4(u3, x1, time_emb)

        output = self.outconv(u4)
        return output