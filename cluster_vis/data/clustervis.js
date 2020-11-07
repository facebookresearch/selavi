// Copyright (c) Facebook, Inc. and its affiliates.

let clusterData = {"kinetics": {"clusters": [[]], "metaclasses": {}}, "vggsound": {"clusters": [[]], "metaclasses": {}}};
let entropy = [];
let sortIndex = {};
let currDataset = "kinetics";
const graphColors = {
    'people': '#2980b9',
    'nature': '#2ecc71',
    'animals': '#1abc9c',
    'sports': '#f39c12',
    'home': '#c0392b',
    'tools': '#8e44ad',
    'music': '#ec407a',
    'vehicle': '#34495e',
    'others': '#7f8c8d'};

function loadVisible() {
    const top = $(document).scrollTop(), bottom = top + $(window).height();
    let results = $('#results > div');
    for (let i = 0; i < results.length; i++) {
        let div = results.eq(i);
        const y1 = div.position().top, y2 = y1 + div.height();
        if (y1 > bottom || y2 < top) {
            if (div.children().length === 2)
              div.children().eq(1).remove();
            continue;
        }
        if (div.children().length === 1)
            $('<iframe>', {'src': div.data('video'), 'width': '100%', 'height': '100%'}).appendTo(div);
    }
}

function makeIndex(index) {
    $('#index').html('');
    let count = index.length;
    for (let i = 0; i < count; i++) {
        $('<a>', {
            text: 'Cluster ' + index[i],
            href: '#welcome-title'
        }).click(function () {
            showCluster(index[i]);
        }).appendTo('#index');
    }
}

function parseParams() {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const id = urlParams.get('id');
    if (id === null)
        return 0;
    return id;
}

function onSortChange() {
    let val = $('#sortSelect').children("option:selected").val();
    makeIndex(sortIndex[val.toLowerCase()]);
}

function showDatasetCluster(dataset, cluster) {
    if (currDataset !== dataset) {
        $("#datasetSelect").val(dataset);
        switchDataset();
    }
    showCluster(cluster);
}

function showRandomCluster(dataset) {
    let nc = clusterData[dataset].clusters.length;
    showDatasetCluster(dataset, Math.floor(Math.random() * nc)); 
}

function showCluster(id) {
    $('#results').html('');
    let classes = {};
    $.each(clusterData[currDataset].clusters[id], function (i, video) {
        let url = 'https://www.youtube.com/embed/' + video[0] + '?start=' + video[1] + '&end=' + video[2];
        $('<div class="result">')
            .data('video', url)
            .append($('<p>').html(video[3]))
            .appendTo('#results');
        classes[video[3]] = (classes[video[3]] || 0) + 1;
    });
    let items = Object.keys(classes).map(function(key) {
        return [key, classes[key]];
    });
    items.sort(function(first, second) {
        return second[1] - first[1];
    });
    let chartLabels = [];
    let datasets = [];
    let categories = Object.keys(graphColors);
    for (let i=0; i<categories.length; i++) {
        datasets.push({
                minBarLength: 2,
                label: categories[i],
				backgroundColor: graphColors[categories[i]],
				borderColor: graphColors[categories[i]],
				borderWidth: 1,
                data: []
            });
    }
    $.each(items, function(index, className) {
        let cat = clusterData[currDataset].metaclasses[className[0]];
        chartLabels.push(className[0]);
        for (let i=0; i<categories.length; i++) {
            if (categories[i] === cat) {
                datasets[i].data.push(className[1]);
            }else{
                datasets[i].data.push(NaN);
            }
        }
    });
    if (window.clusterBarChart)
        window.clusterBarChart.destroy();
    let ctx = document.getElementById('chart-canvas').getContext('2d');
    window.clusterBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: 'ground truth label distribution in this cluster'
            },
            scales: {
                xAxes: [{
                    stacked: true
                }],
                yAxes: [{
                    stacked: true
                }]
            }
        }
    });

    $('#cluster-title').text($("#datasetSelect option:selected").text() +' Cluster '+id+' (entropy: ' + entropy[id].toFixed(3) + ')');
    loadVisible();
}

function computeEntropy(id) {
    let classes = {};
    $.each(clusterData[currDataset].clusters[id], function (i, video) {
        classes[video[3]] = (classes[video[3]] || 0) + 1;
    });
    let entropy = 0.0;
    $.each(classes, function(className, count) {
        let p = count / clusterData[currDataset].clusters[id].length;
        entropy -= p * Math.log2(p);
    });
    return entropy;
}

function addSortIndex(values, name, fac=1) {
    let items = [];
    $.each(values, function (index, value) {
        items.push([index, value*fac]);
    });
    items.sort(function(first, second) {
        return second[1] - first[1];
    });
    sortIndex[name] = items.map(function(item) {return item[0];});
}

function switchDataset() {
    currDataset = $('#datasetSelect').children("option:selected").val();
    let numClusters = clusterData[currDataset].clusters.length;
    entropy = new Array(numClusters);
    let normalIndex = new Array(numClusters);
    for (let i = 0; i < numClusters; i++) {
        entropy[i] = computeEntropy(i);
        normalIndex[i] = i;
    }
    sortIndex["normal"] = normalIndex;
    addSortIndex(entropy, "entropy", -1);
    showCluster(parseParams());
    onSortChange();
}

$( document ).ready(function() {
    $(window).scroll(loadVisible)
    clusterData["kinetics"] = getKineticsClusterData();
    clusterData["vggsound"] = getVGGSoundClusterData();
    switchDataset();

    // add toggle button for gt classes
    $("#cluster-tags").toggle();
    $(".tag-toggle").click(function(){
        $("#tag-toggle-button").toggleClass('collapsed');
        $("#cluster-tags").animate({
            height: 'toggle'
        });
        loadVisible();
    });

    // add toggle button for welcome message
    $(".welcome-toggle").click(function(){
        $("#welcome-toggle-button").toggleClass('collapsed');
        $(".welcome-message").animate({
            height: 'toggle'
        });
    });
});
