%% MTE 546 Data Accquisition Code
% University of Waterloo MTE 546
% Written by Eugene Li Dec 2019
close all; clear all; clc;
%% Device Configuration
%If this is operating correctly the board name should appear
devices = daq.getDevices
s = daq.createSession('ni')

%Add analog inputs for each sensor being read
%Each channel corresponds to the AI on the ELVIS
addAnalogInputChannel(s, 'Dev1', 0, 'Voltage') % thermocouple TopLeft
addAnalogInputChannel(s, 'Dev1', 1, 'Voltage') % thermocouple TopRight
addAnalogInputChannel(s, 'Dev1', 2, 'Voltage') % thermocouple BottomLeft
addAnalogInputChannel(s, 'Dev1', 3, 'Voltage') % thermocouple BottomRight
addAnalogInputChannel(s, 'Dev1', 4, 'Voltage') % IR short TopLeft
addAnalogInputChannel(s, 'Dev1', 5, 'Voltage') % IR short TopRight
addAnalogInputChannel(s, 'Dev1', 6, 'Voltage') % IR long RightTop
addAnalogInputChannel(s, 'Dev1', 7, 'Voltage') % IR long MediumBottom
s.Rate = 20;
%% Sensor Reads

% Debugging Tools
% Do a single sensor read
% data = s.inputSingleScan 

% Read for a second
% [data,time] = s.startForeground;
% plot(time,data); 

% Continuious Background Read
% figure;
% s.DurationInSeconds = 5;
% lh = addlistener(s,'DataAvailable', @(src,event) plot(event.TimeStamps, event.Data));
% s.NotifyWhenDataAvailableExceeds = 8000;
% s.startBackground();
% s.wait()
% delete(lh);

% Normal Operation
% Read for a set period of time
s.DurationInSeconds = 300;
[data, time] = s.startForeground;

%voltage conversions

%{

short range: 10.583031/(data(i,)-0.02725546)
medium range: 24.334172/(data(i,)+0.030560244)
long range: 49.89042/(data(i,)-0.046704563)

%}

distances = zeros(6000,4); %cm
temperatures = zeros(6000,4); %degrees C
for i = 1:6000
    distances(i,1) = 10.583031/(data(i,5)-0.02725546); %convert sensor 1 to distances
end
for i = 1:6000
    distances(i,2) = 10.583031/(data(i,6)-0.02725546); %convert sensor 2 to distances
end
for i = 1:6000
    distances(i,3) = 49.89042/(data(i,7)-0.046704563); %convert sensor 3 to distances
end
for i = 1:6000
    distances(i,4) = 49.89042/(data(i,8)-0.046704563); %convert sensor 4 to distances
end
for i = 1:4
    for j = 1:6000
        temperatures(j,i) = (data(j,i)-1.25)/0.005; %convert thermocouple V to T
    end
end

figure;
plot(time, distances);
legend('Short Left', 'Short Right', 'Long Top', 'Long Bottom');
figure;
plot(time, temperatures);
legend('Top Left', 'Top Right', 'Bottom Left', 'Bottom Right');
