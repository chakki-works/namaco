function getMarkupText(data){
    var dic = {"PERSON": "person", "Person": 'person', "PER": "person",
               "LOCATION": "location", "Location": "location", "LOC": "location",
               "ORG": "organization", "Organization": 'organization', "ORG": "organization",
               "MISC": "misc",
               "FACILITY": "facility", "Facility": "facility",
               "DATE": "date", "Date": "date",
               "Artifact": "artifact",
               "Count": "count",
               "Event": "event",
               "Money": "money",
               "Percentage": "percentage",
               "Time": "time"
               };

    var raw_text = data["text"];
    var markup_text = "";
    var endOffset = 0;
    for (var i = 0; i < data["entities"].length; i++) {
        var entity = data["entities"][i];
        var type = entity["type"];
        var text = entity["text"];
        var beginOffset = entity["beginOffset"];
        markup_text += raw_text.slice(endOffset, beginOffset);
        markup_text += '<small class="axlabel ' + dic[type] + '">' + text + '</small>';
        endOffset = entity["endOffset"];
    }
    markup_text += raw_text.slice(endOffset, raw_text.length);

    return markup_text;
}

$(function () {
	$("#submit").on("click", function (e) {
        var val = $("textarea#sentence").val();
        console.log("value"+val);
        $.post("/", {"sent": val},
           function(data, status){
               console.log(data);
               var text = getMarkupText(JSON.parse(data));
               $(".message-body").html(text);
               console.log(text);
        });
        return true;
    });
});