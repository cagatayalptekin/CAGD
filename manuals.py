
mans = {
    "create": [
        ["t", "Create"],
        ["c", ">_create point [name] [x] [y]"], ["bl"],
        ["c", ">_create curve [name]"],
        ["c", ">_create curve [name] with [point] ..."],
        ["l", " ... unlimited points"], ["bl"],
        ["c", ">_create shape [shape]"],
        ["c", "  + option: radius [value]"],
        ["c", "  + option: at [x] [y]"],
        ["c", "  + option: named [name]"],
        ["c", "  + [shape] = bezier_circle, n-gon, n-m-star"], ["bl"],
        ["n", "- note -"],
        ["n", "creates a new point at (x/y)"],
        ["n", "creates a new curve that is either blank"],
        ["n", "  or filled with specified points"]],
    "list": [
        ["t", "List"],
        ["c", ">_list [type]"],
        ["l", " + [type] = curves, points, saves,"],
        ["l", "            settings, splines, defaults"], ["bl"],
        ["l", ">_list curves ([type])"],
        ["l", " + [type] = table, extra"],
        ["l", ">_list curves [type] sorted by [sort]"],
        ["l", " + [sort] = name, spline, nr # not working"], ["bl"],  # TODO
        ["l", ">_list points ([type])"],
        ["l", " + [type] = table, extra"],
        ["l", ">_list points [type] sorted by [sort]"],
        ["l", " + [sort] = name, x, y, in"],
        ["l", "   + 'in' is nr of curves"], ["bl"],
        ["c", ">_list [type] [name]"],
        ["l", " + [type] = curve, point"], ["bl"],
        ["c", ">_list curve [name] [type]"],
        ["c", " + [type] = names, table"], ["bl"],
        ["n", "- note -"],
        ["n", "lists all of one type"],
        ["n", "  or details about specified object"],
        ["n", "listing a curve gives all points"],
        ["n", "  using the nr in the curve"],
        ["n", "  adding 'names' gives the global"],
        ["n", "  name of the points instead"]],
    "add": [
        ["t", "Add"],
        ["c", ">_add to [curve] [x] [y]"],
        ["l", " + new point (x/y)"], ["bl"],
        ["c", ">_add [point] to [curve]"],
        ["l", " + existing point"], ["bl"],
        ["n", "- note -"],
        ["n", "adds a new or existing"],
        ["n", "  point to a curve"]],
    "edit": [
        ["t", "Edit"],
        ["h", "Edit Curve"],
        ["c", ">_edit curve [name] spline [value]"],
        ["l", " + available spline listed via >_list splines"],
        ["c", ">_edit curve [name] color [value]"],
        ["l", " + color is any string supported by matplotlib"],
        ["c", ">_edit curve [name] point_connection [value]"],
        ["l", " + point_connection = true, false"], ["bl"],
        ["c", ">_edit curve [name] remove point [nr]"],
        ["c", ">_edit curve [name] remove point [nr] & delete"],
        ["c", ">_edit curve [name] remove [name]"],
        ["c", ">_edit curve [name] remove [name] & delete"], ["bl"],
        ["c", ">_edit curve [name] switch [nr] [nr]"], ["bl"],
        ["c", ">_edit curve [name] move point [nr] to [position]"], ["bl"],
        ["c", ">_edit curve [name] reverse points"], ["bl"],
        ["c", ">_edit curve [name] replace point [nr] with [name]"],
        ["c", ">_edit curve [name] replace point [nr] with [x] [y]"],
        ["c", ">_edit curve [name] replace [name] with [name]"],
        ["c", ">_edit curve [name] replace [name] with [x] [y]"], ["bl"],
        ["h", "Edit Point"],
        ["c", ">_edit curve [name] point [nr] [x] [y]"], ["bl"],
        ["c", ">_edit point [name] [x] [y]"],
        ["c", ">_edit point [name] color [value]"],
        ["c", ">_edit point [name] marker [value]"], ["bl"],
        ["h", "Note"],
        ["n", "edits a point or a curve in various ways"],
        ["n", "to check the nr of a point in a curve"],
        ["n", "  use '>_list curve [name]'"],
        ["n", "'& delete' removes the point from the curve"],
        ["n", "  and deletes it globally as well"],
        ["n", "'>_edit curve [name] remove ...':"],
        ["n", "  + point [nr] removes just that one points"],
        ["n", "  + [point] removes all points if duplicates exists"],
        ["n", "  + & delete deletes the point globally and"],
        ["n", "      therefore from all other curves as well"],
        ["n", "markers of points:"],
        ["n", "  - simple ones can be included in the color"],
        ["n", "  - complex ones can be used via >_edit ...marker..."]],
    "change": [
        ["t", "Change"],
        ["c", ">_change resolution [value]"],
        ["l", " + resolution of most curves"], ["bl"],
        ["c", ">_change speed [value]"],
        ["l", " + speed of animations"], ["bl"],
        ["c", ">_change smoothness [value]"],
        ["l", " + resolution of chaikin curves"], ["bl"],
        ["c", ">_change auto_update [value]"],
        ["l", " + [value] = true, false"],
        ["l", " + en-&dis-ables automatic repaint after editing"], ["bl"],
        ["n", "- note -"],
        ["n", "changes settings"]],
    "copy": [
        ["t", "Copy"],
        ["c", ">_copy [type] [name] [new_name]"],
        ["c", ">_copy [type] [name] [new_name] new points"],
        ["l", " + [type] = curve, point"], ["bl"],
        ["n", "- note -"],
        ["n", "creates a copy with a new name"],
        ["n", "new points also creates new points"]],
    "save": [
        ["t", "Save"],
        ["c", ">_save as [name]"], ["bl"],
        ["n", "- note -"],
        ["n", "saves current scene"],
        ["n", "scenes can be loaded"],
        ["n", "  using '>_load' ..."]],
    "load": [
        ["t", "Load"],
        ["c", ">_load from [name]"], ["bl"],
        ["c", ">_load additional [name]"],
        ["l", " + current objects remain"], ["bl"],
        ["n", "- note -"],
        ["n", "load opens a saves scene"],
        ["n", "with 'additional' the current scene remains"],
        ["n", "to see saves use '>_list saves'"]],
    "label": [
        ["t", "Label"],
        ["c", ">_label set [state] all"],
        ["c", ">_label set [state] all [type]"],
        ["l", " + [type] = points, curves"], ["bl"],
        ["c", ">_label set [state] [type] [name]"],
        ["l", " + [type] = point, curve"], ["bl"],
        ["c", ">_label set [state] points of [curve]"], ["bl"],
        ["n", "- note -"],
        ["n", "[state] = hidden, visible"],
        ["n", "sets the visibility of labels "]],
    "delete": [
        ["t", "Delete"],
        ["c", ">_delete [type] [name]"],
        ["l", " + [type] = curve, point"], ["bl"],
        ["c", ">_delete unused points"], ["bl"],
        ["c", ">_delete curve [name] & points"], ["bl"],
        ["n", "- note -"],
        ["n", "deletes a point or a curve"],
        ["n", "'unused points' removes all points"],
        ["n", "  currently not part of any curve"],
        ["n", "'& points' removes the curve and"],
        ["n", "  additionally all its unique points"],
        ["n", "  that are not used by any other curve"]],
    "animate": [
        ["t", "Animate"],
        ["c", ">_animate [curve]"],
        ["c", ">_animate [curve] speed [value]"], ["bl"],
        ["n", "- note -"],
        ["n", "animates a curve step by step"],
        ["n", "  ! only works for Bezier"],
        ["n", "speed is the time in seconds"],
        ["n", "  minimum time (may take longer)"], ["bl"],
        ["n", "default speed can be set"],
        ["n", "  to change >_change speed"],
        ["n", "  to see current >_list settings"]],
    "repaint": [
        ["t", "Repaint"],
        ["c", ">_repaint"], ["bl"],
        ["n", "- note -"],
        ["n", "refreshes the scene"]],
    "modify": [
        ["t", "Modify"],
        ["c", ">_modify [type] [name] move [x] [y]"],
        ["l", " + [type] = point, curve"], ["bl"],
        ["c", ">_modify [type] [name] rotate around [x] [y] [type] [value]"],
        ["c", ">_modify [type] [name] rotate around point [name] [type] [value]"],
        ["c", " + [type] = deg, pi, rad"],
        ["l", " + [type] = point, curve"], ["bl"],
        ["c", ">_modify [type] [name] scale [value] from [x] [y]"],
        ["c", ">_modify [type] [name] scale [value] from point [name]"],
        ["l", " + [type] = point, curve"], ["bl"],
        ["c", ">_modify curve [name] integrate [curve]"], ["bl"],
        ["n", "- note -"],
        ["n", "can move and scale points"],
        ["n", "can move, scale and rotate curves"],
        ["n", "can combine two curves into one"]],
    "rename": [
        ["t", "Rename"],
        ["c", ">rename [type] [name] [new name]"],
        ["l", " + [type] = point, curve"], ["bl"],
        ["n", "- note -"],
        ["n", "renames points and curves"]],
    "reset": [
        ["t", "Reset"],
        ["c", ">rename setting"], ["bl"],
        ["n", "- note -"],
        ["n", "resets all settings to default"]]
}
