function [avgH, avgData] = plotAverage(handleOrData, avgPoints, varargin)
%PLOTAVERAGE plots an average line into a plot (and more)
%
% SYNOPSIS: [avgH, avgData] = plotAverage(handleOrData, avgPoints, parameterName, parameterValue, ...)
%
% INPUT handleOrData: handle to figure or axes of the plot to average. Can
%           be vectors of figures or of axes handles. In a figure with
%           multiple subplots, the average is calculated for each subplot
%           individually.
%           Alternatively, provide a cell array with {x1,y1,x2,y2...},
%           where xi/yi are vectors of different data sets. With the latter
%           form, a plot is generated with figure,plot(x1,y1,x2,y2...).
%           Optional. If empty, plotAverage calls gcf to find the current
%           figure.
%		avgPoints: points on the x-axis (or y-axis, see below) where the
%           average is to be calculated.
%           If empty, the points are selected by locally clustering data
%           points and robustly averaging of the position within each
%           cluster. This works best if the data on the corresponding axes
%           indeed cluster into more or less evenly spaced clusters. If
%           this is not the case, it is probably better to input avgPoints.
%           If avgPoints is a scalar N, the axis is split into N equally
%           spaced points between the minimum and the maximum of the data
%           (excluding the minimum and maximum).
%           Note: If you want to specify separate avgPoints for each of the
%           axes handles passed to plotAverage, pass avgPoints as a cell
%           array.
%
%       plotAverage supports the following parameterName/parameterValue
%           pairs
%		addErrorBars: if 1, error bars are added, if 0, not. Default: 1
%		horzAvg: if 1, average is calculated horizotally (along x) instead
%           of vertically. Default: 0.
%		interpMethod: interpolation method for estimating data values in
%           between support points. See 'help interp1' for supported
%           methods. Default: 'linear'.
%           Use interpMethod='hist' if you want to take the average of 
%           all points in the vicinity of the data (good for scattered data
%           points)
%		plot2NewFigure: if 1, average is plotted in separate figure. If 0,
%           average is plotted on top of the individual data lines. If 2
%           (or an axes handle), the average lines of all the plots are
%           collected in the same figure. Default: 0.
%       useRobustMean: if 1, the robust mean is taken (discarding outliers)
%           for the average curve. If 0, the normal mean is used.
%           Default: 1.
%       plotSEM: if 1, SEM, if 0, the standard deviation is plotted.
%           Default: 1
%
% OUTPUT avgH: handle(s) to average line, plus errorbar handle if
%            applicable
%        avgData: cell array with [x,y,err,n] array of x-values, y-values,
%            standard deviation (not std of the mean) of the average line
%            and number of inlier lines for each data-containing axes.
%            Divide err by sqrt(n) for SEM.
%
% REMARKS (1) This function only works for 2D plots (it ignores axes where
%             the View is not set to [0,90]
%         (2) Since the function looks for axes children of type 'line', it
%             won't work for e.g. bar plots. Also, if you have added error
%             bars with errorbar (instead of myErrorbar), the error bars
%             are included in the averaging, and you will get unexpected
%             results.
%
%
% created with MATLAB ver.: 7.10.0.59 (R2010a) on Mac OS X  Version: 10.6.2 Build: 10C540
%
% created by: jonas
% DATE: 26-Jan-2010
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TEST INPUT

% set defaults
opt = struct(...
    'addErrorBars',true,...
    'horzAvg',false,...
    'interpMethod','linear',...
    'plot2NewFigure',0,...
    'useRobustMean',true,...
    'plotSEM',true);

% find all axes handles, plot data to new figure if necessary
if nargin < 1 || isempty(handleOrData)
    handleOrData = gcf;
end
% class:cell is data, class:double is handle
if isa(handleOrData,'cell')
    if ~isEven(length(handleOrData))
        error('Data needs to be supplied in x.y pairs, e.g. {x1,y1,x2,y2,...}.')
    end
    % plot a new figure
    figure;
    plot(handleOrData{:});
    handleOrData = gca;
end
% loop through handles to get list of axes handles. Skip improper handles
ahList = [];
for ih = 1:length(handleOrData)
    if ishandle(handleOrData(ih))
        % Assume it's a 3D plot if the view is not standard 2D
        if strcmp(get(handleOrData(ih),'type'),'axes') && all(get(handleOrData(ih),'View')==[0,90])
            ahList = [ahList;handleOrData(ih)];
        elseif strcmp(get(handleOrData(ih),'type'),'figure')
            chH = get(handleOrData(ih),'Children');
            % rm legends
            legendIdx = strcmp('legend',get(chH,'Tag'));
            ahList = [ahList;chH(~legendIdx)]; %#ok<*AGROW>
        end
    end
end
if isempty(ahList)
    error('no valid axes handles found in handleOrData or children thereof')
end

% check for other optional inputs
if nargin < 2
    avgPoints = [];
end

if ~isEven(length(varargin))
    error('options must be specified as parameter name/parameter value pairs')
end
for i=1:2:length(varargin)
    opt.(varargin{i}) = varargin{i+1};
end

% turn off robutsMean-warning
oldWarn = warning;
warning off ROBUSTMEAN:INSUFFICIENTDATA

%% CALCULATE AVERAGE

nAh = length(ahList);
data(1:nAh) = struct('xData',[],'yData',[],'avgPoints',avgPoints,'ahIn',num2cell(ahList),'ahOut',[],'avgData',[]);

for ia = nAh:-1:1 % count down in case we remove entries
    % find data in axes
    chH = get(data(ia).ahIn,'Children');
    % remove errorBars, not-lines
    chH(~strcmp('line',get(chH,'Type')) | ismember(get(chH,'Tag'),{'errorBar';'avg'})) = [];
    if isempty(chH)
        % if no valid children, discard axes
        data(ia) = [];
    else
        % get data
        if length(chH) == 1
            data(ia).xData = {get(chH,'XData')};
            data(ia).yData = {get(chH,'YData')};
        else
            data(ia).xData = get(chH,'XData');
            data(ia).yData = get(chH,'YData');
        end
    end
end

nData = length(data);
if nData < 1
    error('no line plots found in the axes provided')
end


% determine x- (or y-) points for calculating the average
for id = 1:nData
    if isempty(data(id).avgPoints)
        if opt.horzAvg
            % collect y
            pts = cat(2,data(id).yData{:})';
        else
            % collect x
            pts = cat(2,data(id).xData{:})';
        end
        % cluster - keep multiples for averaging
        %pts = unique(pts);
        d = pdist(pts); % follow TMW notation
        Z = linkage(d);
        % cutoff is half the average step size
        % Of course, this could theoretically lead to too wide spacing.
        % Hoewever, if there are many points that overlap REALLY well,
        % robustMean gives a cutoff that is way too low.
        cutoff = mean(diff(unique(pts)))/2;
        clust = cluster(Z,'cutoff',cutoff,'criterion','distance');
        % for every cluster, calculate mean
        tmp = NaN(max(clust),1);
        for c=1:max(clust)
            tmp(c) = robustMean(pts(clust==c));
        end
        % remove NaN, sort
        data(id).avgPoints = sort(tmp(isfinite(tmp)));
    elseif isscalar(data(id).avgPoints)
        if opt.horzAvg
            % collect y
            pts = cat(2,data(id).yData{:})';
        else
            % collect x
            pts = cat(2,data(id).xData{:})';
        end
        % linearly space N points
        data(id).avgPoints = linspace(min(pts),max(pts),data(id).avgPoints+2);
        data(id).avgPoints([1,end]) = [];
    end
    
    % now that we know the location, get the value of the average
    nLines = length(data(id).xData);
    avgTmp = NaN(length(data(id).avgPoints),nLines);
    stdTmp = avgTmp;
    for d = 1:nLines
        % if there are multiple 'abscissa'-points with the same value,
        % interpolation fails. Thus, pick the first point if necessary
        if opt.horzAvg
            xx = data(id).yData{d};yy=data(id).xData{d};
        else
            xx = data(id).xData{d};yy=data(id).yData{d};
        end
        
        if strcmp(opt.interpMethod,'hist')
            % associate points in xx with averagePoints. 
            avgPoints = data(id).avgPoints(:);
            meanDelta = mean(diff(avgPoints));
            avgPoints = [avgPoints-meanDelta/2;avgPoints(end)+meanDelta/2];
            [n,binIdx] = histc(xx,avgPoints);
            goodIdx = binIdx>0;
            
            % use accumarray to get average (yes!)
            if opt.useRobustMean
                avgTmp(:,d) = accumarray(binIdx(goodIdx)',yy(goodIdx)',[],@robustMean);
                stdTmp(:,d) = accumarray(binIdx(goodIdx)',yy(goodIdx)',[],@robustStd);
            else
                avgTmp(:,d) = accumarray(binIdx(goodIdx)',yy(goodIdx)',[],@mean);
                stdTmp(:,d) = accumarray(binIdx(goodIdx)',yy(goodIdx)',[],@std);
            end
        else
        % make unique
        [xx,uidx] = unique(xx);
        yy = yy(uidx);
        % remove NaNs
        anyNaN = isnan(xx) | isnan(yy);
        xx(anyNaN) = [];
        yy(anyNaN) = [];
        % interpolate
        if length(xx)>3 && length(yy)>3
        avgTmp(:,d) = interp1(xx,yy,data(id).avgPoints',opt.interpMethod)';
        end
        end
    end
    if nLines == 1 &&  strcmp(opt.interpMethod,'hist')
        data(id).avgData(:,1) = avgTmp;
        data(id).avgData(:,2) = stdTmp;
        data(id).avgData(:,3) = 1;
    elseif opt.useRobustMean && nLines > 4
        [data(id).avgData(:,1),data(id).avgData(:,2),iid] = robustMean(avgTmp,2);
        ctMat = zeros(size(avgTmp));
        ctMat(iid) = 1;
        data(id).avgData(:,3) = sum(ctMat,2);
    else
        data(id).avgData(:,1) = nanmean(avgTmp,2);
        data(id).avgData(:,2) = nanstd(avgTmp,0,2);
        data(id).avgData(:,3) = nLines;
    end
end

%% PLOT AVERAGE

% open a global figure if necessary, otherwise start the plotting loop
if opt.plot2NewFigure == 2
    %outFh = figure('name','collected averages');
    outAh = axes('nextPlot','add');
end
avgH = zeros(nData,1 + opt.addErrorBars);

for id = 1:nData
    % find out where to plot
    switch opt.plot2NewFigure
        case 0
            data(id).ahOut = data(id).ahIn;
            set(data(id).ahOut,'NextPlot','add');
        case 1
            %outFh = figure;
            data(id).ahOut = axes;
        case 2
            data(id).ahOut = outAh;
        otherwise
            % check whether an axes handle has been supplied
            if ishandle(opt.plot2NewFigure) && strcmp(get(opt.plot2NewFigure,'type'),'axes')
                data(id).ahOut = opt.plot2NewFigure;
            else
                error('unsupported option for plot2newFigure')
            end
    end
    
    % plot
    if opt.addErrorBars
        err = data(id).avgData(:,2);
        if opt.plotSEM
            err = err ./ sqrt(data(id).avgData(:,3));
        end
    end
    if opt.horzAvg
        avgH(id,1) = plot(data(id).ahOut,data(id).avgData(:,1),data(id).avgPoints,'k','LineWidth',2,'Tag','avg');
        if opt.addErrorBars
            errH = myErrorbar(data(id).ahOut,data(id).avgData(:,1),data(id).avgPoints,[err;NaN(length(data(id).avgPoints),1);]);
            delete(errH(1));
            avgH(id,2) = errH(2);
        end
    else
        avgH(id,1) = plot(data(id).ahOut,data(id).avgPoints,data(id).avgData(:,1),'k','LineWidth',2,'Tag','avg');
        if opt.addErrorBars
            avgH(id,2) = myErrorbar(data(id).ahOut,data(id).avgPoints,data(id).avgData(:,1),err);
        end
    end
    
    % set legend name
    set(avgH(id,1),'DisplayName','Average line')
    
end % loop data to plot


%% CLEANUP
warning(oldWarn)
if nargout == 0
    clear avgH
end
if nargout > 1
    for id = nData:-1:1
        avgData{id} = [data(id).avgPoints(:),data(id).avgData];
    end
end
