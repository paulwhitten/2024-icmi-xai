const fs = require('fs');
const path = require('path');
const { program } = require('commander');

program
  .option('-k, --knowledge_base <path>');

program.parse();

const options = program.opts();

console.log("Running with options:", options);

const transform_names = [
        'crossing', 'endpoint', 'fill', 'skel-fill', 'skel', 'thresh',
        'line', 'ellipse', 'circle', 'ellipse-circle', 'chull', 'raw', 'corner'
];

let max_label = 0
let min_label = 0xffffffff


let kb = {
    transform_names: [],
    // hashmap storage for per transform statistics
    //   each element is an array of stats for a class
    stats: {},
    // test label array for confirming all match
    //  each element is a label
    labels: undefined,
    // hashmap storage for test predictions per transform
    //   each element is an array of predictions
    test_preds: {}
}


for (let t_name of transform_names) {
    console.log(`=======${t_name}=======`);
    kb.transform_names.push(t_name);
    let train_preds = JSON.parse(fs.readFileSync(path.join(options.knowledge_base, `${t_name}.train.json`)));
    let train_labels = JSON.parse(fs.readFileSync(path.join(options.knowledge_base, `${t_name}.train-labels.json`)));
    let test_preds = JSON.parse(fs.readFileSync(path.join(options.knowledge_base, `${t_name}.test.json`)));
    let test_labels = JSON.parse(fs.readFileSync(path.join(options.knowledge_base, `${t_name}.test-labels.json`)));

    if (!kb.lables) {
        console.log("setting test labels");
        kb.labels = test_labels;
    } else {
        // check for all label arrays to be set and all equal
        if (test_labels.length === kb.labels.length && test_labels.every((element, index) => element === kb.labels[index])) {
            console.log("labels match");
        } else {
            // the test labels must match otherwise not comparing like data across transforms
            console.log("Error: labels do not match");
            process.exit(-1);
        }
    }

    kb.stats[t_name] = get_confusion_matrix(train_labels, train_preds);
    kb.test_preds[t_name] = test_preds;
}

// build up the predicitons and statistical metrics
let predictions = [];
let label_index = 0;
for (let label of kb.labels) {
    let poll = {
        label: label,
        index: label_index,
        votes: []
    };
    for (let t_name of transform_names) {
        //TODO check for other excluded transforms
        if (t_name != "thresh" && t_name != "raw"  && t_name != "skel-fill") { //RAW

            let trans_prediction = kb.test_preds[t_name][label_index];

            poll.votes.push({
                trans_name: t_name,
                class: trans_prediction,
                stats: kb.stats[t_name]['stats'][trans_prediction],
                contribution: get_contribution(kb.stats[t_name]['stats'][trans_prediction])
            });

            // vote from the transform
            // console.log("label:", label, "transform:", t_name, "prediction:", trans_prediction);

            // stats for transform and class digit zero
            //console.log("zero:", kb.stats[t_name]['stats'][0]);
        }
    }
    predictions.push(poll);
    label_index++;
}

let correct = 0;
let second_choice = 0;
let count = 0;
// determine the results by combining predicitons and weighting
let results = [];
for (let p of predictions) {

    let r = {
        label: p.label,
        index: p.index,
        predictions: [],
        sum_weights: 0,
        vote_tally: []
    };

    let label_ix = min_label;
    while (label_ix <= max_label) {

        // represents the votes for a class
        let c = {
            class: label_ix,
            value: 0.0,
            attributions: []
        }

        if (p && p.votes && p.votes.length > 0) {
            for (let v of p.votes) {
                if (v.class == label_ix) {
                    c.value += v.contribution;
                    c.attributions.push({name: v.trans_name, value: v.contribution});
                }
            };
        } else {
            console.log("invalid p.votes:", p);
        }

        if (c.value > 0.0) {
            c.attributions.sort((a, b) => {
                return b.value - a.value;
            })
            r.vote_tally.push(c);
        }

        label_ix++;
    }

    r.vote_tally.sort((a, b) => {
        return b.value - a.value;
    });

    for (let t of r.vote_tally) {
        r.sum_weights += t.value;
    }
    let action = "selected";
    for (let t of r.vote_tally) {
        t["probability"] = t.value / r.sum_weights;
        let str_prob = t.probability.toPrecision(4).toString();
        let props = "";
        let i = 0;
        for (let a of t.attributions) {
            if (i > 0) {
                props += ", ";
                if (i == t.attributions.length - 1) {
                    props += "and ";
                }
            }
            props += a.name;
            i++
        }
        r.predictions.push(`The digit ${t.class} was ${action} with probability ${str_prob} due to the ${props} properties`);
        action = "an alternative";
    }

    // check accuracy of first choice
    if (r.label == r.vote_tally[0].class) {
        correct++;
    } else {
        // TODO looks for other alternatives
        if (r.label == r.vote_tally[1]) {
            second_choice++;
        }
    }

    count++;
    results.push(r);

}

console.log("Result:", correct / count);
console.log("Result 1st and 2nd choice:", (correct + second_choice) / count, second_choice);

fs.writeFileSync(path.join(options.knowledge_base, "results.json"), JSON.stringify(results, null, 4));
fs.writeFileSync(path.join(options.knowledge_base, "kb.json"), JSON.stringify(kb, null, 4));

function get_confusion_matrix(labels, preds) {
    let max = Math.max(...labels);
    let min = Math.min(...labels);
    if (max_label < max) {
        max_label = max;
    }
    if (min_label > min) {
        min_label = min;
    }

    // init the confusion matrix
    let cm = new Array();
    let row_len =  max - min + 1;
    for (let i = 0; i < row_len; i++) {
        cm.push(new Array(row_len).fill(0));
    }
    // init array counting the actual elements per class
    let actuals = new Array(row_len).fill(0);

    // columns are prediction and rows are actual

    let label_count = 0;
    for (let label of labels) {
        actuals[label] += 1;
        let actual = label -  min;
        let pred = preds[label_count] - min;
        cm[actual][pred] += 1;
        label_count++;
    }

    console.log("confusion matrix");
    console.table(cm);

    let s = new Array(row_len).fill().map(u => ({actual: 0, sum: 0, tp: 0, tn: 0, fp: 0, fn: 0, accuracy: 0, sensitivity: 0, specificity: 0, precision: 0}));

    // todo: condider
    //   Precision = tp / (tp + fp)

    for (let j = 0; j < row_len; j++) {
        for (let i = 0; i < row_len; i++) {
            if (j == i) {
                s[j]["tp"] += cm[j][i];
            } else {
                s[j]["fn"] += cm[j][i];
                s[i]["fp"] += cm[j][i]
            }

            for (let k = 0; k < row_len; k++) {
                if (k != j && k != i) {
                    s[k]["tn"] += cm[j][i];
                }
            }

        }
    }

    label_count = 0;
    for (let actual of actuals) {
        let sum = 0;
        for (let key in s[label_count]) {
            sum += s[label_count][key]
        }
        s[label_count]['actual'] = actual;
        s[label_count]['sum'] = sum;
        label_count++;
    }

    for (let t of s) {
        // number of correct predictions
        t.accuracy = (t.tp + t.tn) / t.sum;
        // probability detecting the class given the class
        t.sensitivity = t.tp / t.actual; // same as recall
        // probability of properly detecting not this class
        t.specificity = t.tn / (t.tn + t.fp);
        // out of all positive predictions, ratio that were correct
        t.precision = t.tp / (t.tp + t.fp);
    }

    console.log("stats");
    console.table(s);

    return {
        stats: s,
        confusion_matrix: cm
    };
}

// TODO: do we want the stats weighted?  perhaps parameterize this
function get_contribution(stats) {
    return stats.accuracy * stats.sensitivity * stats.specificity * stats.precision;
}