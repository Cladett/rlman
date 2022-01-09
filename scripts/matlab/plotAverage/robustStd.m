function s = robustStd(data)
%ROBUSTSTD is a wrapper for robustMean and returns the robust standard deviation
%
% SYNOPSIS: s = robustStd(data)
%
% INPUT data: input data
%
% OUTPUT s : robust standard deviation
%
% REMARKS
%
% created with MATLAB ver.: 7.10.0.499 (R2010a) on Microsoft Windows 7 Version 6.1 (Build 7600)
%
% created by: Jonas Dorn
% DATE: 24-Mar-2010
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dummy,s] = robustMean(data);